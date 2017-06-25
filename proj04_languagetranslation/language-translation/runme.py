import helper 
import problem_unittests as tests

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)

view_sentence_range = (0, 10)
import numpy as np 
unique_words = set([word for word in source_text.split()])
print('dataset stats')
print('roughly the number of unique words: {}'.format(len(unique_words)))

sentences = source_text.split('\n')
print(len(sentences))
word_counts = [len(sentence.split()) for sentence in sentences]
print('number of sentences: {}'.format(len(sentences)))
print('avergae number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('english sentences {} to {}'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('french sentences {} to {}'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
	# soure_id 
	return None, None