import typing

from mlflow_extensions.testing.helper import Modality, inject_openai_client, run_if

if typing.TYPE_CHECKING is True:
    from openai import OpenAi

    from mlflow_extensions.testing.runner import ModelContextRunner


@run_if(modality=Modality.TEXT.value)
@inject_openai_client
def query_text(
    *,
    ctx: "ModelContextRunner",
    client: "OpenAi",
    model: str,
    modality_type: str = None,
    host: str = "0.0.0.0",  # noqa
    port: int = 9989,  # noqa
    repeat_n: int = 5,
):
    count = 0
    # repeat a few times
    while count < repeat_n:
        from typing import List, Literal

        from pydantic import BaseModel

        class ExtractedBody(BaseModel):
            product: str
            languages: Literal["python", "sql", "scala"]
            keywords: List[str]
            strategies: List[str]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": """The Databricks Lakehouse Platform for Dummies is your guide to simplifying 
            your data storage. The lakehouse platform has SQL and performance 
            capabilities - indexing, caching and MPP processing - to make 
            BI work rapidly on data lakes. It also provides direct file access 
            and direct native support for Python, data science and 
            AI frameworks without the need to force data through an 
            SQL-based data warehouse. Find out how the lakehouse platform 
            creates an opportunity for you to accelerate your data strategy.""",
                    }
                ],
                extra_body={"guided_json": ExtractedBody.schema()},
            )
            ctx.add_success(result=response.choices[0].message.content)
        except Exception as e:
            print(e)
            ctx.add_error(error_msg=str(e))
        finally:
            count += 1
