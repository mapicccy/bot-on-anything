# encoding:utf-8
import json
import hmac
import hashlib
import base64
import time
import requests
from urllib.parse import quote_plus
from common import log
from flask import Flask, request, render_template, make_response
from common import const
from common import functions
from config import channel_conf
from config import channel_conf_val
from channel.channel import Channel


class PostChannel(Channel):
    def __init__(self):
        log.info("[Post] started.")

    def startup(self):
        http_app.run(host='0.0.0.0', port=channel_conf(const.POST).get('port'))

    def handle(self, prompt, src, dst):
        context = dict()
        context["type"] = "TEXT"
        context["from_user_id"] = ""
        reply = super().build_reply_content(channel_conf(const.POST).get('prefix').format(src, dst) + prompt, context)
        return reply
         

dd = PostChannel()
http_app = Flask(__name__,)


@http_app.route("/trans", methods=['POST'])
def translate():
    # log.info("[Post] chat_headers={}".format(str(request.headers)))
    # log.info("[Post] chat={}".format(str(request.data)))
    token = request.headers.get('token')
    data = json.loads(request.data)
    if data:
        content = data["text_list"]
        if not content:
            return

        reply = {"translations": []}
        for i in range(len(content)):
            reply["translations"].append({"detected_source_lang": data["source_lang"], "text": dd.handle(content[i], data["source_lang"], data["target_lang"])})

        return reply
    
    return {'ret': 201}

