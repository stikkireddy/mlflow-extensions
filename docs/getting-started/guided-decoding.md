# Guided Decoding

The goal of guided decoding is during the decode step of the generation we can control available tokens by applying
a bias to the output. You can learn more from the outlines paper here https://arxiv.org/abs/2307.09702.

The key here is this is done during generation and provides some degree of gaurantees. This is how function calling
works.

If you want to learn more about this in detail you can read this: https://lmsys.org/blog/2024-02-05-compressed-fsm/ by
the
SGLang team.

## Enabling Guided Decoding

The models that support guided decoding automatically have this feature enabled using
[outlines](https://github.com/dottxt-ai/outlines) as the backend.

## Using Guided Decoding with Pydantic

Pydantic is a great way to validate the input and output of the model. You can use python, classes and fields to
constrain the output of the model into a fixed json schema.

```python
from pydantic import BaseModel

class Data(BaseModel):
  outside: bool
  inside: bool

#  construct your client using the guide in ezdeploy or ezdeploylite
client = OpenAI(base_url=..., api_key=...)
response = client.chat.completions.create(
  model="default",
  messages=[
    {"role": "user", "content": [
                {"type": "text", "text": "Is the image indoors or outdoors?"},
                {
                    "type": "image_url",
                    "image_url": {
                      "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    },
                },
            ],
     }
  ],
  #   if you want to use guided decoding to improve performance and control output
  extra_body={
    "guided_json": Data.schema()
  }
)
```

## Guided Decoding options

There are a few options that you can use to control the output of the model.

1. **guided_json**: This is a pydantic schema that you can use to control the output of the model. This is a json
   schema.
2. **guided_regex**: This is a regex that you can use to control the output of the model. This is a string.
3. **guided_choice**: This is a list of choices that you can use to control the output of the model. This is a list of
   strings.
4. **guided_grammar**: This is context free grammar that you can use to control the output of the model. This is a
   string.