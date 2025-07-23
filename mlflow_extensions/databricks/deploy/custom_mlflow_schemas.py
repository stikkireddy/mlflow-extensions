from mlflow.types import ColSpec, DataType, Schema
from mlflow.types.schema import AnyType, Array, Map, Object, Property

EMBEDDING_MODEL_INPUT_SCHEMA = Schema(
    [
        ColSpec(name="input", type=AnyType(), required=True),
    ]
)

EMBEDDING_MODEL_EXAMPLE = {"input": ["this is a test"]}


CHAT_MODEL_INPUT = Schema(
    [
        ColSpec(name="messages", type=AnyType(), required=True),
        ColSpec(name="temperature", type=DataType.double, required=False),
        ColSpec(name="max_tokens", type=DataType.long, required=False),
        ColSpec(name="stop", type=Array(DataType.string), required=False),
        ColSpec(name="n", type=DataType.long, required=False),
        ColSpec(name="stream", type=DataType.boolean, required=False),
        ColSpec(name="top_p", type=DataType.double, required=False),
        ColSpec(name="top_k", type=DataType.long, required=False),
        ColSpec(name="frequency_penalty", type=DataType.double, required=False),
        ColSpec(name="presence_penalty", type=DataType.double, required=False),
        ColSpec(
            name="tools",
            type=Array(
                Object(
                    [
                        Property("type", DataType.string),
                        Property(
                            "function",
                            Object(
                                [
                                    Property("name", DataType.string),
                                    Property("description", DataType.string, False),
                                    Property(
                                        "parameters",
                                        Object(
                                            [
                                                Property(
                                                    "properties",
                                                    Map(
                                                        Object(
                                                            [
                                                                Property(
                                                                    "type",
                                                                    DataType.string,
                                                                ),
                                                                Property(
                                                                    "description",
                                                                    DataType.string,
                                                                    False,
                                                                ),
                                                                Property(
                                                                    "enum",
                                                                    Array(
                                                                        DataType.string
                                                                    ),
                                                                    False,
                                                                ),
                                                                Property(
                                                                    "items",
                                                                    Object(
                                                                        [
                                                                            Property(
                                                                                "type",
                                                                                DataType.string,
                                                                            )
                                                                        ]
                                                                    ),
                                                                    False,
                                                                ),  # noqa
                                                            ]
                                                        )
                                                    ),
                                                ),
                                                Property(
                                                    "type", DataType.string, False
                                                ),
                                                Property(
                                                    "required",
                                                    Array(DataType.string),
                                                    False,
                                                ),
                                                Property(
                                                    "additionalProperties",
                                                    DataType.boolean,
                                                    False,
                                                ),
                                            ]
                                        ),
                                    ),
                                    Property("strict", DataType.boolean, False),
                                ]
                            ),
                            False,
                        ),
                    ]
                ),
            ),
            required=False,
        ),
        ColSpec(name="tool_choice", type=AnyType(), required=False),
        ColSpec(name="custom_inputs", type=Map(AnyType()), required=False),
    ]
)

CHAT_MODEL_EXAMPLE = {
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "this is a test"}],
        }
    ],
    "temperature": 0.1,
    "max_tokens": 10,
    "stop": ["\n"],
    "n": 1,
    "stream": False,
}
