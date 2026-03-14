/*
 * Copyright 2023-present the original author or authors.
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

package org.springframework.ai.mistralai.api;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Flux;

import org.springframework.ai.mistralai.api.MistralAiApi.ChatCompletion;
import org.springframework.ai.mistralai.api.MistralAiApi.ChatCompletionChunk;
import org.springframework.ai.mistralai.api.MistralAiApi.ChatCompletionMessage;
import org.springframework.ai.mistralai.api.MistralAiApi.ChatCompletionMessage.Role;
import org.springframework.ai.mistralai.api.MistralAiApi.ChatCompletionMessage.TextChunk;
import org.springframework.ai.mistralai.api.MistralAiApi.ChatCompletionRequest;
import org.springframework.ai.mistralai.api.MistralAiApi.Embedding;
import org.springframework.ai.mistralai.api.MistralAiApi.EmbeddingList;
import org.springframework.http.ResponseEntity;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * @author Christian Tzolov
 * @author Thomas Vitale
 * @author Jason Smith
 * @author Nicolas Krier
 * @since 0.8.1
 */
@EnabledIfEnvironmentVariable(named = "MISTRAL_AI_API_KEY", matches = ".+")
class MistralAiApiIT {

	private static final Logger LOGGER = LoggerFactory.getLogger(MistralAiApiIT.class);

	private final MistralAiApi mistralAiApi = MistralAiApi.builder()
		.apiKey(System.getenv("MISTRAL_AI_API_KEY"))
		.build();

	@Test
	void chatCompletionEntity() {
		ChatCompletionMessage chatCompletionMessage = new ChatCompletionMessage("Hello world", Role.USER);
		ResponseEntity<ChatCompletion> response = this.mistralAiApi.chatCompletionEntity(new ChatCompletionRequest(
				List.of(chatCompletionMessage), MistralAiApi.ChatModel.MISTRAL_SMALL.getValue(), 0.8, false));

		assertThat(response).isNotNull();
		assertThat(response.getBody()).isNotNull();
	}

	@EnumSource(value = MistralAiApi.ChatModel.class, names = { "MAGISTRAL_MEDIUM", "MAGISTRAL_SMALL" })
	@ParameterizedTest
	void chatCompletionEntityWithThinkingModel(MistralAiApi.ChatModel chatModel) {
		var chatCompletionRequest = new ChatCompletionRequest(createChatCompletionMessages(), chatModel.getValue());
		var responseEntity = this.mistralAiApi.chatCompletionEntity(chatCompletionRequest);

		assertThat(responseEntity).isNotNull();
		assertThat(responseEntity.getStatusCode().is2xxSuccessful()).isTrue();
		var chatCompletion = responseEntity.getBody();
		assertThat(chatCompletion).isNotNull();
		var choices = chatCompletion.choices();
		assertThat(choices).hasSize(1);
		var chatCompletionMessage = choices.get(0).message();
		var content = chatCompletionMessage.extractContent();
		assertThat(content).contains("Jupiter");
		LOGGER.info("Extracted content: {}", content);
		var thinkingContent = chatCompletionMessage.extractThinkingContent();
		assertThat(thinkingContent).isNotEmpty();
		LOGGER.info("Extracted thinking content: {}", thinkingContent);
	}

	@Test
	void chatCompletionEntityWithSystemMessage() {
		ChatCompletionMessage userMessage = new ChatCompletionMessage(
				"Tell me about 3 famous pirates from the Golden Age of Piracy and why they did?", Role.USER);
		ChatCompletionMessage systemMessage = new ChatCompletionMessage("""
				You are an AI assistant that helps people find information.
				Your name is Bob.
				You should reply to the user's request with your name and also in the style of a pirate.
				""", Role.SYSTEM);

		ResponseEntity<ChatCompletion> response = this.mistralAiApi.chatCompletionEntity(new ChatCompletionRequest(
				List.of(systemMessage, userMessage), MistralAiApi.ChatModel.MISTRAL_SMALL.getValue(), 0.8, false));

		assertThat(response).isNotNull();
		assertThat(response.getBody()).isNotNull();
	}

	@Test
	void chatCompletionStream() {
		ChatCompletionMessage chatCompletionMessage = new ChatCompletionMessage("Hello world", Role.USER);
		Flux<ChatCompletionChunk> response = this.mistralAiApi.chatCompletionStream(new ChatCompletionRequest(
				List.of(chatCompletionMessage), MistralAiApi.ChatModel.MISTRAL_SMALL.getValue(), 0.8, true));

		assertThat(response).isNotNull();
		assertThat(response.collectList().block()).isNotNull();
	}

	@EnumSource(value = MistralAiApi.ChatModel.class, names = { "MAGISTRAL_MEDIUM", "MAGISTRAL_SMALL" })
	@ParameterizedTest
	void chatCompletionStreamWithThinkingModel(MistralAiApi.ChatModel chatModel) {
		var chatCompletionRequest = new ChatCompletionRequest(createChatCompletionMessages(), chatModel.getValue(),
				0.7d, true);
		var chatCompletionChunkFlux = this.mistralAiApi.chatCompletionStream(chatCompletionRequest);

		assertThat(chatCompletionChunkFlux).isNotNull();
		var chatCompletionChunks = chatCompletionChunkFlux.collectList().block();
		assertThat(chatCompletionChunks).isNotNull();
		var hasContent = hasContent(chatCompletionChunks, ChatCompletionMessage::extractContent);
		assertThat(hasContent).isTrue();
		var hasThinkingContent = hasContent(chatCompletionChunks, ChatCompletionMessage::extractThinkingContent);
		assertThat(hasThinkingContent).isTrue();
	}

	@Test
	void embeddings() {
		ResponseEntity<EmbeddingList<Embedding>> response = this.mistralAiApi
			.embeddings(new MistralAiApi.EmbeddingRequest<>("Hello world"));

		assertThat(response).isNotNull();
		assertThat(response.getBody()).isNotNull();
		assertThat(response.getBody().data()).hasSize(1);
		assertThat(response.getBody().data().get(0).embedding()).hasSize(1024);
	}

	private static List<ChatCompletionMessage> createChatCompletionMessages() {
		var systemChatCompletionMessage = new ChatCompletionMessage(
				List.of(new TextChunk("You are a helpful assistant providing accurate short answers.")), Role.SYSTEM);
		var userChatCompletionMessage = new ChatCompletionMessage(
				"What is the first planet of the solar system based on the mass in descending order?", Role.USER);

		return List.of(systemChatCompletionMessage, userChatCompletionMessage);
	}

	private static boolean hasContent(List<ChatCompletionChunk> chatCompletionChunks,
			Function<ChatCompletionMessage, String> extractingContentFunction) {
		return chatCompletionChunks.stream()
			.map(ChatCompletionChunk::choices)
			.flatMap(List::stream)
			.map(ChatCompletionChunk.ChunkChoice::delta)
			.map(extractingContentFunction)
			.filter(Objects::nonNull)
			.anyMatch(Predicate.not(String::isEmpty));
	}

}
