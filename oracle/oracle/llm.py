import ast
import base64
import json
import re
from typing import Any, Dict, List, Literal, Tuple, cast

from attr import dataclass
from livekit.agents import llm
from livekit.agents.llm import ChatChunk, ChoiceDelta, FunctionToolCall
from livekit.agents.llm import LLMStream as _LLMStream
from livekit.agents.llm.chat_context import ChatContext, ImageContent
from livekit.agents.llm.tool_context import (
    FunctionTool,
    RawFunctionTool,
    ToolChoice,
    get_function_info,
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)
from livekit.agents.llm.utils import (
    _strict,
    function_arguments_to_pydantic_model,
    serialize_image,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given, shortuuid
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    AssistantMessage,
    ChatDetails,
    DedicatedServingMode,
    DeveloperMessage,
    FunctionCall,
    FunctionDefinition,
    GenericChatRequest,
    ImageUrl,
    OnDemandServingMode,
    SystemMessage,
    TextContent,
    ToolChoiceAuto,
    ToolChoiceNone,
    ToolChoiceRequired,
    ToolMessage,
    UserMessage,
)
from oci.generative_ai_inference.models import ImageContent as OCIImageContent
from oci.retry import NoneRetryStrategy

from .utils import validate_and_prepare_config

ROLE_USER = "user"
ROLE_SYSTEM = "system"
ROLE_ASSISTANT = "assistant"
ROLE_DEVELOPER = "developer"


@dataclass
class _LLMOptions:
    temperature: NotGivenOr[float]
    top_p: NotGivenOr[float]
    tool_choice: NotGivenOr[ToolChoice]
    parallel_tool_calls: NotGivenOr[bool]
    is_stream: bool
    reasoning_effort: NotGivenOr[Literal["NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH"]]


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        compartment_id: str,
        region: str,
        serving_mode: OnDemandServingMode | DedicatedServingMode,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        is_stream: bool = False,
        reasoning_effort: NotGivenOr[
            Literal["NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH"]
        ] = NOT_GIVEN,
        # max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        # max_retries: NotGivenOr[int] = NOT_GIVEN,
    ):
        # TODO: parameter to enable/disable streaming
        super().__init__()
        config = validate_and_prepare_config(config, region)

        # if is_stream:
        #     raise NotImplementedError("Streaming is not supported yet")

        self._opts = _LLMOptions(
            temperature=temperature,
            top_p=top_p,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            is_stream=is_stream,
            reasoning_effort=reasoning_effort,
        )

        self._client = GenerativeAiInferenceClient(
            config=config,
            # TODO: review a more adequate retry strategy and timeout
            retry_strategy=NoneRetryStrategy(),
            timeout=(10, 60),
        )
        self._compartment_id = compartment_id
        self._serving_mode = serving_mode

    @property
    def model(self) -> str:
        """Get the model name for this LLM instance."""
        if isinstance(self._serving_mode, OnDemandServingMode):
            return self._serving_mode.model_id  # type: ignore

        return self._serving_mode.endpoint_id  # type: ignore

    def prewarm(self) -> None:
        self._client.chat(
            chat_details=ChatDetails(
                compartment_id=self._compartment_id,
                serving_mode=self._serving_mode,
                chat_request=GenericChatRequest(
                    messages=[UserMessage(content=[TextContent(text="Say hello")])],
                    is_stream=False,
                ),
            )
        )

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> _LLMStream:
        extra = {}
        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        if is_given(self._opts.temperature):
            extra["temperature"] = self._opts.temperature

        if is_given(self._opts.top_p):
            extra["top_p"] = self._opts.top_p

        parallel_tool_calls = (
            parallel_tool_calls
            if is_given(parallel_tool_calls)
            else self._opts.parallel_tool_calls
        )
        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice  # type: ignore
        if is_given(tool_choice):
            # TODO: handle tool_choice dict as oci ToolChoiceFunction
            if not isinstance(tool_choice, dict) and tool_choice in (
                "auto",
                "required",
                "none",
            ):
                extra["tool_choice"] = tool_choice

        extra["is_stream"] = self._opts.is_stream
        extra["reasoning_effort"] = self._opts.reasoning_effort

        return LLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=(tools or []),
            conn_options=conn_options,
            extra_kwargs=extra,
        )


class LLMStream(_LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._extra_kwargs = extra_kwargs
        self._llm = llm
        self._tool_call_parser = ToolCallParser()

    async def _run(self) -> None:
        messages = []
        for msg in self._chat_ctx.items:
            if msg.type == "message":
                # TODO: accept other content types, like Image, Audio, Video, etc
                if msg.role == ROLE_ASSISTANT:
                    # TODO: capture and group tool_calls in the last assistant message???
                    messages.append(
                        AssistantMessage(content=[TextContent(text=msg.text_content)])
                    )
                elif msg.role == ROLE_DEVELOPER:
                    messages.append(
                        DeveloperMessage(content=[TextContent(text=msg.text_content)])
                    )
                elif msg.role == ROLE_SYSTEM:
                    messages.append(
                        SystemMessage(content=[TextContent(text=msg.text_content)])
                    )
                elif msg.role == ROLE_USER:
                    user_text: list[TextContent] = [TextContent(text=msg.text_content)]
                    user_media: list[OCIImageContent] = []
                    imgs = [
                        image_content
                        for image_content in msg.content
                        if isinstance(image_content, ImageContent)
                    ]

                    for img in imgs:
                        # from livekit.agents.utils.images import encode, EncodeOptions, ResizeOptions
                        # if isinstance(img.image, VideoFrame):
                        #     image_bytes = encode(
                        #         img.image,
                        #         EncodeOptions(
                        #             format="PNG",
                        #             resize_options=ResizeOptions(width=512, height=512, strategy="scale_aspect_fit"),
                        #         ),
                        #     )
                        serialized_img = serialize_image(img)
                        image_url = (
                            serialized_img.external_url
                            if serialized_img.external_url
                            else f"data:{serialized_img.mime_type or 'image/jpeg'};base64,{base64.b64encode(serialized_img.data_bytes or b'').decode('utf-8')}"
                        )
                        user_media.append(
                            OCIImageContent(
                                image_url=ImageUrl(
                                    url=image_url,
                                    detail=(img.inference_detail or "auto").upper(),
                                )
                            )
                        )
                    messages.append(UserMessage(content=(user_media + user_text)))
                    # messages.append(
                    #     UserMessage(content=[TextContent(text=msg.text_content)])
                    # )

            elif msg.type == "function_call":
                messages.append(
                    AssistantMessage(
                        tool_calls=[
                            FunctionCall(
                                id=msg.call_id, name=msg.name, arguments=msg.arguments
                            )
                        ]
                    )
                )
            elif msg.type == "function_call_output":
                messages.append(
                    ToolMessage(
                        tool_call_id=msg.call_id, content=[TextContent(text=msg.output)]
                    )
                )

        tools = []
        for tool in self._tools:
            if is_raw_function_tool(tool):
                info = get_raw_function_info(tool)
                tools.append(
                    FunctionDefinition(
                        name=info.name,
                        parameters=info.raw_schema,
                    )
                )

            elif is_function_tool(tool):
                model = function_arguments_to_pydantic_model(tool)
                info = get_function_info(tool)
                schema = _strict.to_strict_json_schema(model)
                # schema = model.model_json_schema()
                tools.append(
                    FunctionDefinition(
                        name=info.name,
                        description=info.description or "",
                        parameters=schema,
                    )
                )

        # For llama, inject_dummy_user_message!
        user_messages = [
            msg
            for msg in self._chat_ctx.items
            if hasattr(msg, "role") and msg.role == ROLE_USER
        ]
        if len(user_messages) == 0:
            messages.append(UserMessage(content=[TextContent(text=".")]))

        tool_choice_str = self._extra_kwargs.get("tool_choice", None)
        tool_choice = ToolChoiceAuto()
        if tool_choice_str == "auto":
            tool_choice = ToolChoiceAuto()
        elif tool_choice_str == "required":
            tool_choice = ToolChoiceRequired()
        elif tool_choice_str == "none":
            tool_choice = ToolChoiceNone()

        if not self._tools:
            tool_choice = None

        # TODO: retry strategy
        # TODO: restrict to Generic Chat Request only
        # TODO: implement reasoning_effort, verbosity, max_completion_tokens, response_format,...
        is_stream = self._extra_kwargs.get("is_stream", None)
        reasoning_effort = self._extra_kwargs.get("reasoning_effort", None)
        generic_chat_request = GenericChatRequest(
            service_tier="PRIORITY",
            messages=messages,
            is_stream=self._extra_kwargs.get("is_stream", None),
            temperature=self._extra_kwargs.get("temperature", None),
            top_p=self._extra_kwargs.get("top_p", None),
            is_parallel_tool_calls=self._extra_kwargs.get("parallel_tool_calls", None),
            tools=tools,
            tool_choice=tool_choice,
        )
        if reasoning_effort:
            generic_chat_request.reasoning_effort = reasoning_effort

        chat_response = self._llm._client.chat(
            chat_details=ChatDetails(
                compartment_id=self._llm._compartment_id,
                serving_mode=self._llm._serving_mode,
                chat_request=generic_chat_request,
            )
        )

        if is_stream:
            id = shortuuid()
            if chat_response:
                # TODO: IMPORTANT: converter para async! E consumo do generator tbm
                for event in chat_response.data.events():
                    data = json.loads(event.data)
                    message = data.get("message", None)
                    if message:
                        # TODO: handle {'message': {'role': 'ASSISTANT'}, 'finishReason': 'stop', 'pad': 'aaaaaaaa'} ??
                        texts = (
                            item.get("text", "") for item in message.get("content", [])
                        )
                        text = "".join(texts)

                        tool_calls = []
                        for tool_call in message.get("toolCalls", []):
                            tool_calls.append(
                                FunctionToolCall(
                                    name=tool_call.get("name", ""),
                                    arguments=tool_call.get("arguments", ""),
                                    call_id=tool_call.get("id", ""),
                                )
                            )
                        if text or tool_calls:
                            chat_chunk = ChatChunk(
                                id=id,
                                delta=ChoiceDelta(
                                    content=text,
                                    role=ROLE_ASSISTANT,
                                    tool_calls=tool_calls,
                                ),
                            )
                            self._event_ch.send_nowait(chat_chunk)

        else:
            if chat_response:
                # TODO: handle multiple choices
                assistant_message = cast(
                    AssistantMessage,
                    chat_response.data.chat_response.choices[0].message,
                )
                # TODO: add fallback parser for llama
                tool_calls = []
                if assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        tool_calls.append(
                            FunctionToolCall(
                                name=tool_call.name,
                                arguments=tool_call.arguments,
                                call_id=tool_call.id,
                            )
                        )

                # TODO: handle multiple content
                content = ""
                if assistant_message.content and len(assistant_message.content) > 0:
                    original_message = assistant_message.content[0].text
                    message_tool_calls = self._tool_call_parser.parse(original_message)
                    if len(message_tool_calls) > 0:
                        tool_calls.extend(message_tool_calls)
                    content = self._tool_call_parser.clean_message(original_message)

                chat_chunk = ChatChunk(
                    id=shortuuid(),
                    delta=ChoiceDelta(
                        content=content,
                        role=ROLE_ASSISTANT,
                        tool_calls=tool_calls,
                    ),
                )
                self._event_ch.send_nowait(chat_chunk)


class ToolCallParser:
    def __init__(self):
        self.bracket_pattern = r"\[([^\[\]]+)(?:\]|$)"
        self.function_pattern = r"(\w+)\((.*?)\)(?=\s*(?:,\s*\w+\(|$))"
        self.param_pattern = r"(\w+)\s*=\s*([^,\)]+)"

    def parse_value(self, value_str: str) -> Any:
        value_str = value_str.strip()

        if value_str.lower() == "null" or value_str.lower() == "none":
            return None

        if value_str.lower() == "true":
            return True
        elif value_str.lower() == "false":
            return False

        try:
            if (value_str.startswith('"') and value_str.endswith('"')) or (
                value_str.startswith("'") and value_str.endswith("'")
            ):
                return value_str[1:-1]
            else:
                return ast.literal_eval(value_str)
        except Exception:
            if (value_str.startswith('"') and value_str.endswith('"')) or (
                value_str.startswith("'") and value_str.endswith("'")
            ):
                return value_str[1:-1]
            return value_str

    def parse_parameters(self, params_str: str) -> Dict[str, Any]:
        params = {}
        if not params_str.strip():
            return params

        matches = re.finditer(self.param_pattern, params_str)

        for match in matches:
            param_name = match.group(1)
            param_value = match.group(2)
            params[param_name] = self.parse_value(param_value)

        return params

    def parse_function_call(self, call_str: str) -> Tuple[str, Dict[str, Any]]:
        match = re.match(r"(\w+)\((.*)\)", call_str.strip())
        if not match:
            raise ValueError(f"Invalid function call format: {call_str}")

        func_name = match.group(1)
        params_str = match.group(2)
        params = self.parse_parameters(params_str)

        return func_name, params

    def parse(self, text: str) -> List[Dict[str, Any]]:
        results = []
        bracket_matches = re.finditer(self.bracket_pattern, text)

        for bracket_match in bracket_matches:
            calls_str = bracket_match.group(1)
            function_matches = re.finditer(self.function_pattern, calls_str)

            for func_match in function_matches:
                func_name = func_match.group(1)
                params_str = func_match.group(2)
                params = self.parse_parameters(params_str)

                results.append({"function": func_name, "parameters": params})

        return results

    def clean_message(self, text: str) -> str:
        cleaned = re.sub(self.bracket_pattern, "", text)
        cleaned = re.sub(r"\s+", " ", cleaned)

        lines = [line.strip() for line in cleaned.split("\n")]
        lines = [line for line in lines if line]

        cleaned = "\n".join(lines)
        return cleaned.strip()
