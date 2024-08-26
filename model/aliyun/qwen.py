# encoding:utf-8

from model.model import Model
from config import model_conf, common_conf_val
from common import const
from common import log
from dashscope import Generation, ImageSynthesis

import requests
import openai
import time
import json

user_session = dict()
total_used_tokens = 0

# aliyun对话模型API (Qwen)
class QwenModel(Model):
    def __init__(self):
        self.api_key = model_conf(const.ALIYUN).get('api_key')
        self.api_base = model_conf(const.ALIYUN).get('api_base')
        self.proxy = model_conf(const.ALIYUN).get('proxy')
        self.enable_search = model_conf(const.ALIYUN).get('enable_search')
        self.temperature = model_conf(const.ALIYUN).get('temperature')
        self.client = openai.OpenAI(api_key=self.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    def reply(self, query, context=None):
        # acquire reply content
        if not context or not context.get('type') or context.get('type') == 'TEXT':
            log.info("[QWen] query={}".format(query))
            from_user_id = context['from_user_id']
            clear_memory_commands = common_conf_val('clear_memory_commands', ['#清除记忆'])
            if query in clear_memory_commands:
                Session.clear_session(from_user_id)
                return '记忆已清除'

            if from_user_id == "":
                Session.clear_session(from_user_id)

            new_query = Session.build_session_query(query, from_user_id)
            log.debug("[QWen] session query={}".format(new_query))

            # if context.get('stream'):
            #     # reply in stream
            #     return self.reply_text_stream(query, new_query, from_user_id)

            reply_content = self.reply_text(new_query, from_user_id, 0)
            #log.debug("[CHATGPT] new_query={}, user={}, reply_cont={}".format(new_query, from_user_id, reply_content))
            return reply_content

        elif context.get('type', None) == 'IMAGE_CREATE':
            return self.create_img(query, 0)

    def reply_text(self, query, user_id, retry_count=0):
        try:
            response = Generation.call(
                model= model_conf(const.ALIYUN).get("model") or "qwen-turbo",  # 对话模型的名称
                messages=query,
                temperature=self.temperature,  # 值在[0,2)之间，越大表示回复越具有不确定性
                enable_search=self.enable_search,
                top_p=1,
                result_format="message",
                frequency_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
                presence_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            )
            reply_content = response.output.choices[0]["message"]["content"]
            used_token = response.usage.total_tokens
            global total_used_tokens
            total_used_tokens = total_used_tokens + used_token
            log.info("[QWen] reply={}", reply_content)
            log.info("[QWen] total used tokens={}".format(total_used_tokens))
            if reply_content:
                # save conversation
                Session.save_session(query, reply_content, user_id, used_token)
            return reply_content
        except Exception as e:
            # unknown exception
            log.exception(e)
            Session.clear_session(user_id)
            return "出错了，请再问我一次吧。reason: {}".format(response.message)


    def reply_text_stream(self, query, new_query, user_id, retry_count=0):
        try:
            res = openai.Completion.create(
                model="text-davinci-003",  # 对话模型的名称
                prompt=new_query,
                temperature=0.9,  # 值在[0,1]之间，越大表示回复越具有不确定性
                #max_tokens=4096,  # 回复最大的字符数
                top_p=1,
                frequency_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
                presence_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
                stop=["\n\n\n"],
                stream=True
            )
            return self._process_reply_stream(query, res, user_id)

        except Exception as e:
            # unknown exception
            log.exception(e)
            Session.clear_session(user_id)
            return "出错了，请再问我一次吧"


    def _process_reply_stream(
            self,
            query: str,
            reply: dict,
            user_id: str
    ) -> str:
        full_response = ""
        for response in reply:
            if response.get("choices") is None or len(response["choices"]) == 0:
                raise Exception("Aliyun API returned no choices")
            if response["choices"][0].get("finish_details") is not None:
                break
            if response["choices"][0].get("text") is None:
                raise Exception("Aliyun API returned no text")
            if response["choices"][0]["text"] == "<|endoftext|>":
                break
            yield response["choices"][0]["text"]
            full_response += response["choices"][0]["text"]
        if query and full_response:
            Session.save_session(query, full_response, user_id)


    def create_img(self, query, retry_count=0):
        try:
            log.info("[QWen] image_query={}".format(query))
            """
            response = ImageSynthesis.call(model=ImageSynthesis.Models.wanx_v1,
                                           prompt=query,    #图片描述
                                           n=1,             #每次生成图片的数量
                                           size="1024*1024"   #图片大小,可选有 256x256, 512x512, 1024x1024
                                          )
            """
            url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
            headers = {
                'X-DashScope-Async': 'enable',
                'Authorization': 'Bearer {}'.format(self.api_key),
                'Content-Type': 'application/json'
            }
            data = {
                "model": "wanx-v1",
                "input": {
                    "prompt": query,
                    "negative_prompt": "低分辨率、错误、最差质量、低质量、jpeg 伪影、丑陋、重复、病态、残缺、超出框架、多余的手指、变异的手、画得不好的手、画得不好的脸、突变、变形、模糊、脱水、不良的解剖结构、 比例不良、多余肢体、克隆脸、毁容、总体比例、畸形肢体、缺臂、缺腿、多余手臂、多余腿、融合手指、手指过多、长脖子、用户名、水印、签名"
                },
                "parameters": {
                    "style": "<auto>",
                    "size": "1024*1024",
                    "n": 1
                }
            }
            r = requests.post(url, headers=headers, data=json.dumps(data))
            response = r.json()
            task = response["output"]['task_id']
            status = response['output']['task_status']
            image_url = ''
            while status != "FAILED" or status != "UNKNOWN":
                qurl = "https://dashscope.aliyuncs.com/api/v1/tasks/{}".format(task)
                headers = {
                    'Authorization': 'Bearer {}'.format(self.api_key),
                }
                r = requests.get(qurl, headers=headers)
                response = r.json()
                status = response['output']['task_status']
                if status == "SUCCEEDED":
                    image_url = response['output']['results'][0]['url']
                    break

                time.sleep(0.001)

            log.info("[QWen] image_url={}".format(image_url))
            return [image_url]
        except Exception as e:
            log.exception(e)
            return None


class Session(object):
    @staticmethod
    def build_session_query(query, user_id):
        '''
        build query with conversation history
        e.g.  [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
        :param query: query content
        :param user_id: from user id
        :return: query content with conversaction
        '''
        session = user_session.get(user_id, [])
        if len(session) == 0:
            system_prompt = model_conf(const.ALIYUN).get("character_desc", "")
            system_item = {'role': 'system', 'content': system_prompt}
            session.append(system_item)
            user_session[user_id] = session
        user_item = {'role': 'user', 'content': query}
        session.append(user_item)
        return session

    @staticmethod
    def save_session(query, answer, user_id, used_tokens=0):
        max_tokens = model_conf(const.ALIYUN).get('conversation_max_tokens')
        max_history_num = model_conf(const.ALIYUN).get('max_history_num', None)
        if not max_tokens or max_tokens > 4000:
            # default value
            max_tokens = 1000
        session = user_session.get(user_id)
        if session:
            # append conversation
            gpt_item = {'role': 'assistant', 'content': answer}
            session.append(gpt_item)

        if used_tokens > max_tokens and len(session) >= 3:
            # pop first conversation (TODO: more accurate calculation)
            session.pop(1)
            session.pop(1)

        if max_history_num is not None:
            while len(session) > max_history_num * 2 + 1:
                session.pop(1)
                session.pop(1)

    @staticmethod
    def clear_session(user_id):
        user_session[user_id] = []

