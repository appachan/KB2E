import logging.config

_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simpleFormatter': {
            'format': '[%(asctime)s] %(levelname)-8s %(module)-18s %(funcName)-10s %(lineno)4s: %(message)s'
        }
    },
    'handlers': {
        'consoleHandler': {
            'level': 'DEBUG',
            'formatter': 'simpleFormatter',
            'class': 'logging.StreamHandler',
        },
        'fileHandler': {
            'level': 'INFO',
            'formatter': 'simpleFormatter',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logging.log',
            'maxBytes': 1000000,
            'backupCount': 3,
            'encoding': 'utf-8',
        }
    },
    'loggers': {
        '': {
            'handlers': ['consoleHandler', 'fileHandler'],
            'level': "DEBUG",
        }
    }
}

logging.config.dictConfig(_config)