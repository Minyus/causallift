import logging

from IPython.display import display
import numpy as np
import pandas as pd

from .utils import *  # NOQA

log = logging.getLogger(__name__)


class ModelForTreatedOrUntreated:
    def __init__(self, treatment_val=1.0):
        assert treatment_val in {0.0, 1.0}
        self.treatment_val = treatment_val
        self.treatment_label = "treated" if treatment_val else "untreated"

    def fit(self, args, df_):

        assert isinstance(df_, pd.DataFrame)
        treatment_val = self.treatment_val

        if args.verbose >= 2:
            log.info("\n\n## Model for Treatment = {}".format(treatment_val))

        df = df_.query("{}=={}".format(args.col_treatment, treatment_val)).copy()

        X_train = df.xs("train")[args.cols_features]
        y_train = df.xs("train")[args.col_outcome]
        X_test = df.xs("test")[args.cols_features]
        y_test = df.xs("test")[args.col_outcome]

        model = initialize_model(args, model_key="uplift_model_params")

        if args.enable_ipw and (args.col_propensity in df.xs("train").columns):
            propensity = df.xs("train")[args.col_propensity]

            # avoid propensity near 0 or 1 which will result in too large weight
            if propensity.min() < args.min_propensity and args.verbose >= 2:
                log.warning(
                    "[Warning] Propensity scores below {} were clipped.".format(
                        args.min_propensity
                    )
                )
            if propensity.max() > args.max_propensity and args.verbose >= 2:
                log.warning(
                    "[Warning] Propensity scores above {} were clipped.".format(
                        args.max_propensity
                    )
                )
            propensity.clip(
                lower=args.min_propensity, upper=args.max_propensity, inplace=True
            )

            sample_weight = (
                (1 / propensity) if treatment_val == 1.0 else (1 / (1 - propensity))
            )

            model.fit(X_train, y_train, sample_weight=sample_weight)
        
        elif args.enable_weighting and (args.col_weight in df.xs("train").columns):
            sample_weight = df.xs("train")[args.col_weight]
            model.fit(X_train, y_train, sample_weight=sample_weight)

        else:
            model.fit(X_train, y_train)

        best_estimator = (
            model.best_estimator_ if hasattr(model, "best_estimator_") else model
        )
        estimator_params = best_estimator.get_params()
        if "steps" in estimator_params:
            best_estimator = estimator_params["steps"][-1][1]
            estimator_params = best_estimator.get_params()

        if args.verbose >= 3:
            log.info(
                "### Best parameters of the model trained using samples "
                "with observational Treatment: {} \n {}".format(
                    treatment_val, estimator_params
                )
            )

        if args.verbose >= 2:
            if hasattr(estimator_params, "feature_importances_"):
                fi_df = pd.DataFrame(
                    estimator_params.feature_importances_.reshape(1, -1),
                    index=["feature importance"],
                )
                log.info(
                    "\n### Feature importances of the model trained using samples "
                    "with observational Treatment: {}".format(treatment_val)
                )
                display(fi_df)
            else:
                log.info("## Feature importances not available.")

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        score_original_treatment_df = score_df(
            y_train, y_test, y_pred_train, y_pred_test, average="binary"
        )
        if args.verbose >= 3:
            log.info(
                "\n### Outcome estimated by the model trained using samples "
                "with observational Treatment: {}".format(treatment_val)
            )
            display(score_original_treatment_df)

        model_dict = dict(model=model, eval_df=score_original_treatment_df)
        return model_dict

    def predict_proba(self, args, df_, models_dict):
        model = models_dict[self.treatment_label]["model"]

        cols_features = args.cols_features

        X_train = df_.xs("train")[cols_features]
        X_test = df_.xs("test")[cols_features]

        y_pred_train = model.predict_proba(X_train)[:, 1]
        y_pred_test = model.predict_proba(X_test)[:, 1]

        return concat_train_test(args, y_pred_train, y_pred_test)

        # X = df_[cols_features]
        # y_pred = model.predict_proba(X)[:, 1]
        # return y_pred

    def simulate_recommendation(self, args, df_, models_dict):

        model = models_dict[self.treatment_label]["model"]
        score_original_treatment_df = models_dict[self.treatment_label]["eval_df"]

        treatment_val = self.treatment_val
        verbose = args.verbose

        df = df_.query("{}=={}".format(args.col_recommendation, treatment_val)).copy()

        X_train = df.xs("train")[args.cols_features]
        y_train = df.xs("train")[args.col_outcome]
        X_test = df.xs("test")[args.cols_features]
        y_test = df.xs("test")[args.col_outcome]

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        score_recommended_treatment_df = score_df(
            y_train, y_test, y_pred_train, y_pred_test, average="binary"
        )
        if verbose >= 3:
            log.info(
                "\n### Simulated outcome of samples recommended to be treatment: {} by the uplift model:".format(
                    treatment_val
                )
            )
            display(score_recommended_treatment_df)

        out_df = pd.DataFrame(index=["train", "test"])
        out_df["# samples chosen without uplift model"] = score_original_treatment_df[
            ["# samples"]
        ]
        out_df[
            "observed conversion rate without uplift model"
        ] = score_original_treatment_df[["observed conversion rate"]]
        out_df[
            "# samples recommended by uplift model"
        ] = score_recommended_treatment_df[["# samples"]]
        out_df[
            "predicted conversion rate using uplift model"
        ] = score_recommended_treatment_df[["predicted conversion rate"]]

        out_df["predicted improvement rate"] = (
            out_df["predicted conversion rate using uplift model"]
            / out_df["observed conversion rate without uplift model"]
        )

        return out_df


class ModelForTreated(ModelForTreatedOrUntreated):
    def __init__(self, *posargs, **kwargs):
        kwargs.update(treatment_val=1.0)
        super().__init__(*posargs, **kwargs)


class ModelForUntreated(ModelForTreatedOrUntreated):
    def __init__(self, *posargs, **kwargs):
        kwargs.update(treatment_val=0.0)
        super().__init__(*posargs, **kwargs)


def model_for_treated_fit(*posargs, **kwargs):
    return ModelForTreated().fit(*posargs, **kwargs)


def model_for_treated_predict_proba(*posargs, **kwargs):
    return ModelForTreated().predict_proba(*posargs, **kwargs)


def model_for_treated_simulate_recommendation(*posargs, **kwargs):
    return ModelForTreated().simulate_recommendation(*posargs, **kwargs)


def model_for_untreated_fit(*posargs, **kwargs):
    return ModelForUntreated().fit(*posargs, **kwargs)


def model_for_untreated_predict_proba(*posargs, **kwargs):
    return ModelForUntreated().predict_proba(*posargs, **kwargs)


def model_for_untreated_simulate_recommendation(*posargs, **kwargs):
    return ModelForUntreated().simulate_recommendation(*posargs, **kwargs)


def bundle_treated_and_untreated_models(treated_model, untreated_model):
    models_dict = dict(treated=treated_model, untreated=untreated_model)
    return models_dict
