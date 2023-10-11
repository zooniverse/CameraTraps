# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
A class to get configurations for each instance of the API.
"""
import os

from server_api_config import API_INSTANCE_NAME, CALLER_ALLOW_LIST

class AppConfig:
    """Wrapper around the Azure App Configuration client"""

    def __init__(self):
        self.api_instance = API_INSTANCE_NAME

        # setup the known caller allowList using delimited ENV var string
        self.allowlist = CALLER_ALLOW_LIST.split(';')

    def get_allowlist(self):
        return self.allowlist

