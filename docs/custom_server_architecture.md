# Custom Server Architecture

This document goes over how the custom server frameoworks like vLLM, sglang and ray serve work in the context 
mlflow-extensions. It tries to explain the architecture of the custom server frameworks and how they are integrated
into the default scoring server for mlflow and databricks deployments.

## Custom Supported Server Frameworks

1. vLLM
2. sglang
3. ray serve
4. ollama

## Custom Pyfunc Implementation

All of these frameworks typically run in their own processes and have their own command to boot up. Every framework is 
wrapped under a generic pyfunc to proxy /invocation requests to the custom server. The custom pyfunc also serializes
and deserializes an httpx request and response object between the client (openai, httpx, sglang) to custom server implementation.

Here is a high level diagram of the architecture:

![mlflow-extensions-server.png](static%2Fmlflow-extensions-server.png)


Mlflow pyfuncs are split up into three main parts, `__init__`, `load_context` and `predict`. The constructor in this case 
is a very simple implementation. The `load_context` function is where the interesting bits happen.


### Understanding `load_context`

Before we dive into `load_context` it is important to understand that when a model spawns in the scoring server it 
spawns multiple workers, in the case of mlflow it will use gunicorn as the server framework. Each worker will have its 
own instance of the model. This is important because some large models wont have enough resources if you are spawning 
more than one instance of the model. So instead `load_context` on the worker will call a "filelock" (using the filelock library) 
to lock various worker process using a filesystem file lock. This allows the various gunicorn processes to coordinate on 
a single leader which will spawn instances of the model serving framework (vLLM, sglang, etc). This is important because 
the model serving framework will have its own process and will be able to handle multiple requests at once. The last part 
of the `load_context` function is to run a separate process to ensure the server framework is healthy, and up and 
running. This is important, sometimes the SOTA server frameworks can be a bit unstable, throw segfaults, crash, etc. 
This health check restarts the server if it is down. 

Some room for improvements here would be to ensure that when the model is not in a healthy state that the requests should 
let the clients know that the model is not healthy. This is important because the serving endpoint may be up but the model
may be down and in a recovering state.

### Understanding `predict`

The goal for predict is to proxy the request to the custom server. This is done by serializing the httpx Request into a 
json string and sending it to the custom server. The custom server will then deserialize the json string back into an 
httpx Request object and then run the request through the custom server. The custom server will then the proper response
back to the scoring server. The scoring server will then serialize the response back into a json string and send it back 
to the client. The client will then deserialize the json string back into an httpx Response object. This is the biggest 
overhead in the system. The serialization and deserialization of the httpx Request and Response objects. This is important 
to note because the custom server will have to be able to handle the serialization and deserialization of the httpx efficiently.
We have not currently measured the overhead of this.

The trade off though mlflow scoring server does not support dynamic/adaptive batching. The client will have to handle 
batching requests to the scoring server. Most models like llms, transformers, etc will have a batching strategy to load 
batches of text, images, etc into the model and this framework will allow you to do that. So this is the trade off that 
you will have to make when using the custom server frameworks. (e2e latency over dynamic batching)

### Compatability Clients

The `mlflow_extensions.serving.compat` has a few clients that are compatible with the custom server frameworks. These clients 
are `OpenAI`, `ChatOpenAI`, `RuntimeEndpoint` (sglang), and standard httpx sync and async clients. These clients are 
used to serialize and deserialize requests to the custom server. 


### Request flow

In this the pyfunc wrapper refers to the gunicorn workers/scoring server (mlflow specific workers)

1. Compatability OpenAI Client sends a chat completion request
2. Wrapper client intercepts the request and serializes the request into a json string
3. Wrapper client sends the request to the pyfunc hosted in model serving
4. Pyfunc wrapper deserializes the json string back into an httpx Request object
5. Pyfunc wrapper runs the request through the custom server
6. Custom server batches/executes the request
7. Custom server sends the response back to the pyfunc wrapper
8. Pyfunc wrapper serializes the response back into a json string
9. Client deserializes the json string back into an httpx Response object
10. Compatability OpenAI Client receives the response and transforms it into a chat completion response

This is the detailed step by step process of how the request flows through the system. It all happens very quick but 
this is the indirection that is happening in the system.
