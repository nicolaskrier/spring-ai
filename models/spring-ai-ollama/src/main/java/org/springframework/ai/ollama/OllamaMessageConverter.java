/*
 * Copyright 2025-2025 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.ollama;

import com.fasterxml.jackson.core.type.TypeReference;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.content.Media;
import org.springframework.ai.content.MediaContent;
import org.springframework.ai.ollama.api.OllamaApi;
import org.springframework.ai.util.json.JsonParser;
import org.springframework.core.convert.converter.Converter;
import org.springframework.lang.NonNull;
import org.springframework.util.CollectionUtils;

import java.util.Base64;
import java.util.List;
import java.util.stream.Stream;

/**
 * Converter from chat model message to stream of Ollama API messages.
 *
 * @author Nicolas Krier
 */
public class OllamaMessageConverter implements Converter<Message, Stream<OllamaApi.Message>> {

	@Override
	public final @NonNull Stream<OllamaApi.Message> convert(@NonNull Message message) {
		return switch (message.getMessageType()) {
			case USER -> fromUserMessage(message);
			case SYSTEM -> fromSystemMessage(message);
			case ASSISTANT -> fromAssistantMessage(message);
			case TOOL -> fromToolMessage(message);
		};
	}

	protected Stream<OllamaApi.Message> fromUserMessage(Message message) {
		if (message instanceof MediaContent mediaContent) {
			List<String> images = null;

			if (!CollectionUtils.isEmpty(mediaContent.getMedia())) {
				images = mediaContent.getMedia()
						.stream()
						.map(Media::getData)
						.map(this::fromMediaData)
						.toList();
			}

			return Stream.of(OllamaApi.Message.builder(OllamaApi.Message.Role.USER)
					.content(message.getText())
					.images(images)
					.build());
		}
		else {
			throw new IllegalArgumentException("Message class not supported for user: " + message.getClass());
		}
	}

	protected Stream<OllamaApi.Message> fromSystemMessage(Message message) {
		return Stream.of(OllamaApi.Message.builder(OllamaApi.Message.Role.SYSTEM)
				.content(message.getText())
				.build());
	}

	protected Stream<OllamaApi.Message> fromAssistantMessage(Message message) {
		if (message instanceof AssistantMessage assistantMessage) {
			List<OllamaApi.Message.ToolCall> toolCalls = null;

			if (!CollectionUtils.isEmpty(assistantMessage.getToolCalls())) {
				toolCalls = assistantMessage.getToolCalls()
						.stream()
						.map(this::fromToolCall)
						.toList();
			}

			return Stream.of(OllamaApi.Message.builder(OllamaApi.Message.Role.ASSISTANT)
					.content(message.getText())
					.toolCalls(toolCalls)
					.build());
		}
		else {
			throw new IllegalArgumentException("Message class not supported for assistant: " + message.getClass());
		}
	}

	protected Stream<OllamaApi.Message> fromToolMessage(Message message) {
		if (message instanceof ToolResponseMessage toolResponseMessage) {
			return toolResponseMessage.getResponses()
					.stream()
					.map(ToolResponseMessage.ToolResponse::responseData)
					.map(data -> OllamaApi.Message.builder(OllamaApi.Message.Role.TOOL)
							.content(data)
							.build());
		}
		else {
			throw new IllegalArgumentException("Message class not supported for tool: " + message.getClass());
		}
	}

	private String fromMediaData(Object mediaData) {
		if (mediaData instanceof byte[] bytes) {
			return Base64.getEncoder().encodeToString(bytes);
		}
		else if (mediaData instanceof String text) {
			return text;
		}
		else {
			throw new IllegalArgumentException("Unsupported media data type: " + mediaData.getClass().getSimpleName());
		}
	}

	private OllamaApi.Message.ToolCall fromToolCall(AssistantMessage.ToolCall toolCall) {
		var toolCallFunction = new OllamaApi.Message.ToolCallFunction(toolCall.name(),
				JsonParser.fromJson(toolCall.arguments(), new TypeReference<>() {
				}));
		return new OllamaApi.Message.ToolCall(toolCallFunction);
	}

}
