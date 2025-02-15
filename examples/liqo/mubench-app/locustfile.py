#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from locust import FastHttpUser, TaskSet, between, constant_throughput
from faker import Faker
import datetime
fake = Faker()

def index(l):
    l.client.get("/s0")

class UserBehavior(TaskSet):

    def on_start(self):
        index(self)

    tasks = {index: 1}

class WebsiteUser(FastHttpUser):
    tasks = [UserBehavior]
    #wait_time = between(1, 10)
    wait_time = constant_throughput(1)
