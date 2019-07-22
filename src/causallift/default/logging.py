def conf_logging_():
    conf_logging = \
        {'disable_existing_loggers': False,
         'formatters': {
             'json_formatter': {
                 'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                 'format': '[%(asctime)s] %(message)s',
             },
             'simple': {
                 'format': '[%(asctime)s] %(message)s',
             },
         },
         'handlers': {'console': {'class': 'logging.StreamHandler',
                                  'formatter': 'simple',
                                  'level': 'INFO',
                                  'stream': 'ext://sys.stdout'}},
         'loggers': {
             'anyconfig': {
                 'handlers': ['console'],
                 'level': 'WARNING',
                 'propagate': False,
             },
             'kedro.io': {
                 'handlers': ['console'],
                 'level': 'WARNING',
                 'propagate': False,
             },
             'kedro.pipeline': {
                 'handlers': ['console'],
                 'level': 'WARNING',
                 'propagate': False,
             },
             'causallift': {
                 'handlers': ['console'],
                 'level': 'INFO',
                 'propagate': False,
             },
         },
         'root': {
             'handlers': ['console'],
             'level': 'WARNING',
         },
         'version': 1}
    return conf_logging
