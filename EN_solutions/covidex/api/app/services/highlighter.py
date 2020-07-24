from typing import List, Tuple

import spacy
import torch
import transformers

from app.settings import settings


class Highlighter:
    def __init__(self):

        self.device = torch.device(settings.highlight_device)

        print('Loading tokenizer...')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            'monologg/biobert_v1.1_pubmed', do_lower_case=False)
        print('Loading model...')
        self.model = transformers.AutoModel.from_pretrained(
            'monologg/biobert_v1.1_pubmed')
        self.model.to(self.device)

        print('Loading sentence tokenizer...')
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))

        self.highlight_token = '[HIGHLIGHT]'
        self.max_paragraph_length = 10000

    def text_to_vectors(self, text: str):
        """Converts a text to a sequence of vectors, one for each subword."""
        text_ids = torch.tensor(
            self.tokenizer.encode(text, add_special_tokens=True))
        text_ids = text_ids.to(self.device)

        text_words = self.tokenizer.convert_ids_to_tokens(text_ids)[1:-1]

        states = []
        for i in range(1, text_ids.size(0), 510):
            text_ids_ = text_ids[i: i + 510]
            text_ids_ = torch.cat([text_ids[0].unsqueeze(0), text_ids_])

            if text_ids_[-1] != text_ids[-1]:
                text_ids_ = torch.cat(
                    [text_ids_, text_ids[-1].unsqueeze(0)])

            with torch.no_grad():
                state, _ = self.model(text_ids_.unsqueeze(0))
                state = state[0, 1:-1, :]
            states.append(state)
        state = torch.cat(states, axis=0)
        return text_words, state

    def similarity_matrix(self, vector1, vector2):
        """Compute the cosine similarity matrix of two vectors of same size.

        Args:
            vector1: A torch vector of size N.
            vector2: A torch vector of size N.

        Returns:
            A similarity matrix of size N x N.
        """
        vector1 = vector1 / torch.sqrt((vector1 ** 2).sum(1, keepdims=True))
        vector2 = vector2 / torch.sqrt((vector2 ** 2).sum(1, keepdims=True))
        return (vector1.unsqueeze(1) * vector2.unsqueeze(0)).sum(-1)

    def adjust_highlights(self, original_text: str, highlighted_text: str):
        """
        Adjusts highlights to ignore extra spaces introduced by the tokenization process.
        Iterates over highlighted and original text simultaneously to compute character positions.

        Returns:
            Original text with highlight tokens inserted in correct locations.
        """

        original_idx = 0
        hlt_idx = 0
        highlighted_original_text = ''

        while original_idx < len(original_text) and hlt_idx < len(highlighted_text):
            if original_text[original_idx] == highlighted_text[hlt_idx]:
                highlighted_original_text += original_text[original_idx]
                original_idx += 1
                hlt_idx += 1
            elif highlighted_text[hlt_idx:hlt_idx+len(self.highlight_token)] == self.highlight_token:
                # Insert highlight token into original text
                highlighted_original_text += self.highlight_token
                hlt_idx += len(self.highlight_token)
            elif highlighted_text[hlt_idx:hlt_idx+5] == "[UNK]":
                hlt_idx += 5
                if hlt_idx >= len(highlighted_text):
                    break
                # Original text may have multiple spaces and characters to form [UNK] token
                highlighted_next = highlighted_text[hlt_idx]
                while original_text[original_idx] != highlighted_next:
                    highlighted_original_text += original_text[original_idx]
                    original_idx += 1
            elif original_text[original_idx] == " ":
                # Handle extra spaces in original text
                highlighted_original_text += original_text[original_idx]
                original_idx += 1
            else:
                # Handle extra spaces in tokenized text
                hlt_idx += 1

        # Append remaining characters
        if original_idx < len(original_text):
            highlighted_original_text += original_text[original_idx:]

        return highlighted_original_text

    def highlight_paragraph(self, query_state, para_state,
                            para_words, original_paragraph) -> List[Tuple[int]]:
        '''Returns the start and end character positions of highlighted sentences'''

        if not original_paragraph:
            return []

        sim_matrix = self.similarity_matrix(
            vector1=query_state, vector2=para_state)

        # Select the two highest scoring words in the sim_matrix.
        _, word_positions = torch.topk(
            sim_matrix.max(0)[0], k=2, largest=True, sorted=False)
        word_positions = word_positions.tolist()

        # Append a special highlight token to top-scoring words.
        for kk in word_positions:
            para_words[kk] += self.highlight_token

        tagged_paragraph = self.tokenizer.convert_tokens_to_string(para_words)
        tagged_paragraph = self.adjust_highlights(original_paragraph, tagged_paragraph)

        tagged_sentences = [
            sent.string.strip()
            for sent in self.nlp(tagged_paragraph[:self.max_paragraph_length]).sents]

        new_paragraph = []
        highlights = []
        last_pos = 0
        for sent in tagged_sentences:
            if self.highlight_token in sent:
                sent = sent.replace(self.highlight_token, '')
                highlights.append((last_pos, last_pos + len(sent)))

            new_paragraph.append(sent)
            last_pos += len(sent) + 1

        return highlights

    def highlight_paragraphs(self, query: str,
                             paragraphs: List[str]) -> List[List[Tuple[int]]]:
        """Highlight sentences in a list of paragraph based on their
        similarity to the query.

        Args:
            query: A query text.
            paragraphs: A list of paragraphs

        Returns:
            all_highlights: A list of lists of tuples, where the elements of
                the tuple denote the start and end positions of the segments
                to be highlighted.
        """

        query_words, query_state = self.text_to_vectors(text=query)

        new_paragraphs = []
        all_highlights = []
        for paragraph in paragraphs:
            para_words, para_state = self.text_to_vectors(text=paragraph)
            highlights = self.highlight_paragraph(
                query_state=query_state,
                para_state=para_state,
                para_words=para_words,
                original_paragraph=paragraph)
            all_highlights.append(highlights)
        return all_highlights
