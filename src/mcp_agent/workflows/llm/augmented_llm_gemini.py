import base64
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from pydantic import BaseModel, ConfigDict, Field
import mcp
from mcp_agent.logging.logger import get_logger
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MCPMessageParam,
    ModelT,
    RequestParams,
    ProviderToMCPConverter,
)
from mcp.types import (
    ModelPreferences,
    TextResourceContents,
    BlobResourceContents,
    TextContent,
    ImageContent,
    EmbeddedResource,
    CallToolRequest,
    CallToolRequestParams,
)

import google.genai
import google.genai.types
from google.genai.types import (
    FunctionResponse,
    FunctionCall,
    ToolListUnion,
    Schema,
    Tool,
    Content,
    Part,
    Blob,
    FileData,
    GenerateContentResponse,
    GenerateContentConfig,
    UserContent,
    Candidate,
    FunctionDeclaration,
)

GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"


Message = GenerateContentResponse
MessageParam = Content

# class MessageParam(BaseModel):
#     contents: Union[ContentListUnion, ContentListUnionDict]
#     config: Optional[GenerateContentConfigOrDict] = None


class GeminiAugmentedLLM(AugmentedLLM[MessageParam, Message]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, type_converter=MCPGeminiTypeConverter, **kwargs)

        self.provider = "Gemini"
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

        self.model_preferences = self.model_preferences or ModelPreferences(
            costPriority=0.3,
            speedPriority=0.4,
            intelligencePriority=0.3,
        )
        self.default_request_params = self.default_request_params or RequestParams(
            model=GEMINI_DEFAULT_MODEL,
            modelPreferences=self.model_preferences,
            maxTokens=4096,
            systemPrompt=self.instruction,
            parallel_tool_calls=False,
            max_iterations=10,
            use_history=True,
        )

    @classmethod
    def convert_message_to_message_param(
        cls, message: Message, **kwargs
    ) -> MessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        # return ChatCompletionAssistantMessageParam(
        #     role="assistant",
        #     content=message.content,
        #     tool_calls=message.tool_calls,
        #     audio=message.audio,
        #     refusal=message.refusal,
        #     **kwargs,
        # )
        if message.candidates is None or len(message.candidates) == 0:
            raise ValueError("Message has no candidates")
        elif len(message.candidates) > 1:
            raise NotImplementedError("Message has multiple candidates")

        content = message.candidates[0].content
        if content is None:
            raise ValueError("Message candidate has no content")

        return MessageParam(
            role=content.role,
            parts=content.parts,
            **kwargs,
        )

    async def generate(self, message, request_params: RequestParams | None = None):
        config = self.context.config
        gemini_client = google.genai.Client(api_key=config.gemini.api_key)
        messages: List[MessageParam] = []
        params = self.get_request_params(request_params)

        if params.use_history:
            messages.extend(self.history.get())

        system_prompt = self.instruction or params.systemPrompt

        if isinstance(message, str):
            messages.append(UserContent(message))
        elif isinstance(message, list):
            messages.extend(message)
        else:
            messages.append(message)

        response = await self.aggregator.list_tools()

        available_tools: ToolListUnion = [
            Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=Schema.model_validate(
                            SchemaIgnoreExtra.model_validate(tool.inputSchema)
                        ),
                    )
                ]
            )
            for tool in response.tools
        ]

        responses: List[GenerateContentResponse] = []
        model = await self.select_model(params)

        from rich import print

        print(available_tools)

        for i in range(params.max_iterations):
            config = GenerateContentConfig(
                system_instruction=system_prompt, tools=available_tools
            )
            arguments = {
                "model": params.model,
                "contents": "Hello",
                "config": config,
            }

            executor_result = await self.executor.execute(
                gemini_client.models.generate_content, **arguments
            )
            # executor_result = gemini_client.models.generate_content(**arguments)

            response = executor_result[0]

            self.logger.debug("Gemini ChatCompletion response:", data=response)

            if isinstance(response, BaseException):
                self.logger.error(
                    f"Error: {response}",
                )
                break

            if response.candidates is None or len(response.candidates) == 0:
                break

            responses.append(response)

            coverted_message = self.convert_message_to_message_param(response)
            messages.append(coverted_message)
            candidate = response.candidates[0]

            if candidate.content is not None and candidate.content.parts is not None:
                tool_tasks = [
                    self.execute_tool_call(part.function_call)
                    for part in candidate.content.parts
                    if part.function_call is not None
                ]
                tool_results = await self.executor.execute(*tool_tasks)

                for result in tool_results:
                    if isinstance(result, BaseException):
                        self.logger.error(
                            f"Warning: Unexpected error during tool execution: {result}. Continuing..."
                        )
                        continue
                    if result is not None:
                        messages.append(result)

            if candidate.finish_reason is not None:
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is '{candidate.finish_message}'"
                )
                break

        if params.use_history:
            self.history.set(messages)

        self._log_chat_finished(model=model)

        return responses

    async def generate_str(self, message, request_params: RequestParams | None = None):
        responses = await self.generate(
            message=message,
            request_params=request_params,
        )

        final_text: List[str] = []

        for response in responses:
            content = response.text
            if content is None:
                continue
            final_text.append(content)

        return "\n".join(final_text)

    async def generate_structured(
        self,
        message,
        response_model: type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        import instructor

        response = await self.generate_str(
            message=message,
            request_params=request_params,
        )

        gemini_client = google.genai.Client(api_key=self.context.config.gemini.api_key)
        instructor_client = instructor.from_gemini(
            client=gemini_client,
            mode=instructor.Mode.TOOLS_STRICT,
        )

        params = self.get_request_params(request_params)
        model = await self.select_model(params)

        structured_response = instructor_client.chat.completions.create(
            model=model or GEMINI_DEFAULT_MODEL,
            response_model=response_model,
            messages=[{"role": "user", "content": response}],
        )

        return structured_response

    async def execute_tool_call(
        self,
        tool_call: FunctionCall,
    ) -> MessageParam | None:
        if tool_call.name is None:
            return None

        tool_call_request = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=tool_call.name, arguments=tool_call.args),
        )
        result = await self.call_tool(
            request=tool_call_request, tool_call_id=tool_call.id
        )

        if result.content is None:
            return None

        return MessageParam(
            role="user",
            parts=[
                Part(
                    function_response=FunctionResponse(
                        id=tool_call.id,
                        name=tool_call.name,
                        response=mcp_content_to_dict(content),
                    )
                )
                for content in result.content
            ],
        )


class MCPGeminiTypeConverter(ProviderToMCPConverter[MessageParam, Message]):
    @classmethod
    def from_mcp_message_param(cls, param):
        extras = param.model_dump(exclude={"role", "parts"})
        return MessageParam(
            role=mcp_role_to_gemini_role(param.role),
            parts=[mcp_content_to_gemini_content(param.content)],
            **extras,
        )

    @classmethod
    def to_mcp_message_param(cls, param):
        extras = param.model_dump(exclude={"role", "content"})
        contents = gemini_content_to_mcp_content(param)

        if len(contents) > 1:
            raise NotImplementedError(
                "Multiple content elements in a single message are not supported"
            )

        mcp_content = contents[0]

        if isinstance(mcp_content, EmbeddedResource):
            raise NotImplementedError("EmbeddedResource is not supported")

        return MCPMessageParam(
            role=gemini_role_to_mcp_role(param.role),
            content=mcp_content,
            **extras,
        )

    @classmethod
    def from_mcp_message_result(cls, result):
        return Message(
            candidates=[
                Candidate(
                    content=Content(
                        role=mcp_role_to_gemini_role(result.role),
                        parts=[mcp_content_to_gemini_content(result.content)],
                    )
                )
            ]
        )

    @classmethod
    def to_mcp_message_result(cls, result):
        return super().to_mcp_message_result(result)


def mcp_content_to_dict(
    content: TextContent | ImageContent | EmbeddedResource,
) -> dict[str, Any]:
    if isinstance(content, TextContent):
        return {"text": content.text}
    elif isinstance(content, ImageContent):
        return {"data": content.data, "mimeType": content.mimeType}
    elif isinstance(content, EmbeddedResource):
        return {"resource": content.resource}
    else:
        raise NotImplementedError(f"Unsupported content type: {type(content)}")


def mcp_role_to_gemini_role(role: mcp.types.Role | None) -> str:
    if role == "user":
        return "user"
    elif role == "assistant":
        return "model"
    else:
        raise NotImplementedError(f"Unsupported role: {role}")


def gemini_role_to_mcp_role(role: str | None) -> mcp.types.Role:
    if role == "user":
        return "user"
    elif role == "model":
        return "assistant"
    else:
        raise NotImplementedError(f"Unsupported role: {role}")


def mcp_content_to_gemini_content(
    content: TextContent | ImageContent | EmbeddedResource,
) -> Part:
    if isinstance(content, TextContent):
        return Part(text=content.text)
    elif isinstance(content, ImageContent):
        return Part(
            file_data=FileData(
                file_uri=content.data,
                mime_type=content.mimeType,
            )
        )
    elif isinstance(content, EmbeddedResource):
        if isinstance(content.resource, TextResourceContents):
            return Part(text=content.resource.text)
        elif isinstance(content.resource, BlobResourceContents):
            if content.resource.mimeType == "text/plain":
                return Part(text=content.resource.blob)
            else:
                return Part(
                    inline_data=Blob(
                        data=base64.b64decode(content.resource.blob),
                        mime_type=content.resource.mimeType,
                    )
                )
        else:
            raise NotImplementedError(
                f"Unsupported resource type: {type(content.resource)}"
            )
    else:
        raise NotImplementedError(f"Unsupported content type: {type(content)}")


def gemini_content_to_mcp_content(
    content: MessageParam,
) -> List[TextContent | ImageContent | EmbeddedResource]:
    if content.parts is None:
        return []

    mcp_content = []
    for part in content.parts:
        if part.text is not None:
            return [TextContent(type="text", text=part.text)]
        elif part.file_data is not None:
            if part.file_data.mime_type is None:
                raise ValueError("FileData must have a mime type")
            elif part.file_data.mime_type.startswith("image/"):
                if part.file_data.file_uri is None:
                    raise ValueError("FileData must have a file URI")
                mcp_content.append(
                    ImageContent(
                        type="image",
                        data=part.file_data.file_uri,
                        mimeType=part.file_data.mime_type,
                    )
                )
            else:
                raise NotImplementedError(
                    f"Unsupported mime type: {part.file_data.mime_type}"
                )
        elif part.inline_data is not None:
            raise NotImplementedError("Inline data not supported")
        elif part.video_metadata is not None:
            raise NotImplementedError("Video metadata not supported")
        elif part.executable_code is not None:
            if part.executable_code.code is None:
                raise ValueError("ExecutableCode must have code")
            # !TODO code language not supported
            mcp_content.append(
                TextContent(
                    type="text",
                    text=part.executable_code.code,
                )
            )
        elif part.code_execution_result is not None:
            raise NotImplementedError("Code execution result not supported")

        elif part.function_call is not None:
            raise NotImplementedError("Function call not supported")
        elif part.function_response is not None:
            raise NotImplementedError("Function response not supported")
        else:
            raise NotImplementedError(f"Unsupported content type: {type(content)}")

    return mcp_content


def typed_dict_extras(d: dict, exclude: List[str]):
    extras = {k: v for k, v in d.items() if k not in exclude}
    return extras


# Based on google.genai.types class Schema
# Modified to ignore extra fields
class SchemaIgnoreExtra(BaseModel):
    """Schema that defines the format of input and output data.

    Represents a select subset of an OpenAPI 3.0 schema object.
    """

    example: Optional[Any] = Field(
        default=None,
        description="""Optional. Example of the object. Will only populated when the object is the root.""",
    )
    pattern: Optional[str] = Field(
        default=None,
        description="""Optional. Pattern of the Type.STRING to restrict a string to a regular expression.""",
    )
    # default: Optional[Any] = Field(
    #     default=None, description="""Optional. Default value of the data."""
    # )
    max_length: Optional[int] = Field(
        default=None,
        description="""Optional. Maximum length of the Type.STRING""",
    )
    min_length: Optional[int] = Field(
        default=None,
        description="""Optional. SCHEMA FIELDS FOR TYPE STRING Minimum length of the Type.STRING""",
    )
    min_properties: Optional[int] = Field(
        default=None,
        description="""Optional. Minimum number of the properties for Type.OBJECT.""",
    )
    max_properties: Optional[int] = Field(
        default=None,
        description="""Optional. Maximum number of the properties for Type.OBJECT.""",
    )
    any_of: Optional[list["Schema"]] = Field(
        default=None,
        description="""Optional. The value should be validated against any (one or more) of the subschemas in the list.""",
    )
    description: Optional[str] = Field(
        default=None, description="""Optional. The description of the data."""
    )
    enum: Optional[list[str]] = Field(
        default=None,
        description="""Optional. Possible values of the element of primitive type with enum format. Examples: 1. We can define direction as : {type:STRING, format:enum, enum:["EAST", NORTH", "SOUTH", "WEST"]} 2. We can define apartment number as : {type:INTEGER, format:enum, enum:["101", "201", "301"]}""",
    )
    format: Optional[str] = Field(
        default=None,
        description="""Optional. The format of the data. Supported formats: for NUMBER type: "float", "double" for INTEGER type: "int32", "int64" for STRING type: "email", "byte", etc""",
    )
    items: Optional["Schema"] = Field(
        default=None,
        description="""Optional. SCHEMA FIELDS FOR TYPE ARRAY Schema of the elements of Type.ARRAY.""",
    )
    max_items: Optional[int] = Field(
        default=None,
        description="""Optional. Maximum number of the elements for Type.ARRAY.""",
    )
    maximum: Optional[float] = Field(
        default=None,
        description="""Optional. Maximum value of the Type.INTEGER and Type.NUMBER""",
    )
    min_items: Optional[int] = Field(
        default=None,
        description="""Optional. Minimum number of the elements for Type.ARRAY.""",
    )
    minimum: Optional[float] = Field(
        default=None,
        description="""Optional. SCHEMA FIELDS FOR TYPE INTEGER and NUMBER Minimum value of the Type.INTEGER and Type.NUMBER""",
    )
    nullable: Optional[bool] = Field(
        default=None,
        description="""Optional. Indicates if the value may be null.""",
    )
    properties: Optional[dict[str, "SchemaIgnoreExtra"]] = Field(
        default=None,
        description="""Optional. SCHEMA FIELDS FOR TYPE OBJECT Properties of Type.OBJECT.""",
    )
    property_ordering: Optional[list[str]] = Field(
        default=None,
        description="""Optional. The order of the properties. Not a standard field in open api spec. Only used to support the order of the properties.""",
    )
    required: Optional[list[str]] = Field(
        default=None,
        description="""Optional. Required properties of Type.OBJECT.""",
    )
    title: Optional[str] = Field(
        default=None, description="""Optional. The title of the Schema."""
    )
    type: Optional[google.genai.types.Type] = Field(
        default=None, description="""Optional. The type of the data."""
    )

    model_config = ConfigDict(extra="ignore")
