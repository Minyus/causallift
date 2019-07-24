def conf_logging_():
    conf_logging = \
        {'disable_existing_loggers': False,
         'formatters': {
             'json_formatter': {
                 'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                 'format': '[%(asctime)s|%(name)s|%(funcName)s|%(levelname)s] %(message)s',
             },
             'simple': {
                 'format': '[%(asctime)s|%(name)s|%(levelname)s] %(message)s',
             },
         },
         'handlers': {
             'console': {
                 'class': 'logging.StreamHandler',
                 'formatter': 'simple',
                 'level': 'INFO',
                 'stream': 'ext://sys.stdout',
             },
            'info_file_handler': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'filename': './info.log',
                'maxBytes': 10485760, # 10MB
                'backupCount': 20,
                'encoding': 'utf8',
                'delay': True,
            },
             'error_file_handler': {
                 'class': 'logging.handlers.RotatingFileHandler',
                 'level': 'ERROR',
                 'formatter': 'simple',
                 'filename': './errors.log',
                 'maxBytes': 10485760,  # 10MB
                 'backupCount': 20,
                 'encoding': 'utf8',
                 'delay': True,
             },
         },
         'loggers': {
             'anyconfig': {
                 'handlers': ['console', 'info_file_handler', 'error_file_handler'],
                 'level': 'WARNING',
                 'propagate': False,
             },
             'kedro.io': {
                 'handlers': ['console', 'info_file_handler', 'error_file_handler'],
                 'level': 'WARNING',
                 'propagate': False,
             },
             'kedro.pipeline': {
                 'handlers': ['console', 'info_file_handler', 'error_file_handler'],
                 'level': 'INFO',
                 'propagate': False,
             },
             'kedro.runner': {
                 'handlers': ['console', 'info_file_handler', 'error_file_handler'],
                 'level': 'INFO',
                 'propagate': False,
             },
             'causallift': {
                 'handlers': ['console', 'info_file_handler', 'error_file_handler'],
                 'level': 'INFO',
                 'propagate': False,
             },
         },
         'root': {
             'handlers': ['console', 'info_file_handler', 'error_file_handler'],
             'level': 'INFO',
         },
         'version': 1}
    return conf_logging
