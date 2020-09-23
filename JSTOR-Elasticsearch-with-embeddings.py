import logging
import sys
import re
import xml.etree.ElementTree as ET
import xmltodict

from lxml import etree
import os
import traceback
import elasticsearch
import elasticsearch.helpers
import json
from enum import Enum
from scipy.spatial.distance import cosine

import torch
from transformers import BertTokenizer, BertModel

# CONFIGS
DATASET_DIR = "/scratch/avtale/jstor/letter_fgh" #root directory for files
ES_INDEX_NAME = "jstor_articles_w_contextual_embeddings"
ES_HOST = "agave1.agave.rc.asu.edu:9200"
MAPPINGS_JSON = "data/mapping_beckett_jstor"
LOG_PREFIX = "ace"

ES_AUTH_USER = None
ES_AUTH_PASSWORD = None

ES_DOCUMENT_TYPE = 'article'
ES_CREATE_INDEX = True
ES_TIMEOUT = 60

DEBUG = ''
KEY_SPECIES="species"
KEY_SUBSPECIES="subspecies"


# LOGGING
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger('jstor')
DEBUG_LOG_FILENAME = '{}_debug.log'.format(LOG_PREFIX)
WARN_LOG_FILENAME = '{}_warn.log'.format(LOG_PREFIX)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
warning_handler = logging.FileHandler(WARN_LOG_FILENAME)
warning_handler.setLevel(level=logging.WARNING)
warning_handler.setFormatter(formatter)
debug_handler = logging.FileHandler(DEBUG_LOG_FILENAME)
debug_handler.setLevel(level=logging.DEBUG)
debug_handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(warning_handler)
logger.addHandler(debug_handler)


#HELPER FUNCTIONS
def split_into_sentences(text):
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


#CLASSES
class EmbeddingType(Enum):
    SPECIES = KEY_SPECIES
    SUBSPECIES = KEY_SUBSPECIES

class LanguageError(Exception):
    pass

class XMLParser(object):
    def _process_if_avail(self, d, key, func, target=None):
        try:
            value = d.pop(key)
        except KeyError:
            return
        else:
            if value is not None:
                new_value = func(value)
                if new_value is not None:
                    d[target or key] = new_value

    def _process__article_meta__contrib_group(self, contrib_groups):
        if not isinstance(contrib_groups, list):
            contrib_groups = [contrib_groups]

        for contrib_group in contrib_groups:
            if not isinstance(contrib_group['contrib'], list):
                contrib_group['contrib'] = [contrib_group['contrib']]

            try:
                if isinstance(contrib_group['aff'], str):
                    contrib_group['aff'] = [contrib_group['aff']]
            except KeyError:
                pass

            for i, contrib in enumerate(contrib_group['contrib']):
                try:
                    name = contrib['string-name']
                except KeyError:
                    pass
                else:
                    if isinstance(name, str):
                        name = [name]
                    contrib_group['contrib'][i]['string-name'] = name
                try:
                    if isinstance(contrib['aff'], str):
                        contrib['aff'] = [contrib['aff']]
                except KeyError:
                    pass

        return contrib_groups

    def _process__article_meta__pub_date(self, pub_dates):
        years = None
        if isinstance(pub_dates, dict):
            if isinstance(pub_dates['year'], list):
                return pub_dates['year']
            else:
                return [pub_dates['year']]

        return [pub_date['year'] for pub_date in pub_dates]

    def _process__article_meta(self, article_meta):
        _make_str = lambda d: d.get('#text', None) if isinstance(d, dict) else d

        # Remove duplicates
        for key in ('issue-id', 'issue', 'volume', 'pub-date'):
            if isinstance(article_meta.get(key, ''), list):
                if article_meta[key][0] == article_meta[key][1]:
                    article_meta[key] = article_meta[key][0]

        self._process_if_avail(article_meta, 'pub-date', self._process__article_meta__pub_date, 'year')
        # self._process_if_avail(article_meta, 'contrib-group', self._process__article_meta__contrib_group)
        # self._process_if_avail(article_meta, 'issue-id', _make_str)
        # self._process_if_avail(article_meta, 'issue', _make_str)
        # self._process_if_avail(article_meta, 'volume', _make_str)

        try:
            self._process_if_avail(article_meta['title-group'], 'article-title', _make_str)
        except KeyError:
            pass
        return article_meta

    def _process__journal_meta(self, journal_meta):
        journal_title_str = lambda journal_title: journal_title['#text'] if isinstance(journal_title,
                                                                                       dict) else journal_title
        self._process_if_avail(journal_meta['journal-title-group'], 'journal-title', journal_title_str)
        try:
            journal_meta['journal-title'] = journal_meta.pop('journal-title-group')['journal-title']
        except KeyError:
            pass
        return journal_meta

    def _get_parse_postprocessor(self, article_et):
        etree_str = lambda e: etree.tostring(e, encoding='utf-8', method='text').strip()
        isenglish = lambda x: re.match(x, 'eng?', re.I)

        def postprocessor(path, key, value):
            xpath = '/'.join((path[i][0] for i in range(1, len(path))))

            if key == '@xml:lang' and not isenglish(value):
                raise LanguageError(value)

            if xpath == 'front/article-meta/custom-meta-group/custom-meta/meta-value' and not isenglish(value):
                raise LanguageError(value)

            if value is None:
                return None

            if key in set((
                    '@xmlns:xsi',
                    '@xml:lang',
                    '@xlink:type',
                    '@ext-link-type',
                    '@content-type',
                    '@dtd-version',
                    '@xmlns:oasis',
                    '@xmlns:xlink',
                    '@xmlns:mml',
                    '@xlink:role',
                    '@xlink:title',
                    '@article-type',
                    'fig-count',
                    'equation-count',
                    'table-count',
            )):
                return None

            if re.search('|'.join((
                    '^front/article-meta/kwd-group/x',
                    '^front/article-meta/contrib-group/x',
                    '^front/article-meta/contrib-group/xref',
                    '^front/article-meta/contrib-group/contrib/x',
                    '^front/article-meta/contrib-group/contrib/xref',
                    '^front/article-meta/related-article',
                    '^front/article-meta/product',
                    '^front/article-meta/permissions',
                    '^front/article-meta/custom-meta-group',
                    '^front/article-meta/trans-abstract',
                    '^front/article-meta/title-group/trans-title-group',
                    '^front/article-meta/title-group/trans-title-group/trans-title',
                    '^front/article-meta/article-categories/subj-group/subj-group',
                    '^front/journal-meta/custom-meta-group',
                    '^body',
            )), xpath):
                return None

            if key == 'email':
                if isinstance(value, dict):
                    value = value['#text']
                return key, value

            if key in set((
                    'page-count',
                    'ref-count',
            )):
                return key, int(value['@count'])

            if xpath == 'front/article-meta/self-uri' and not isinstance(value, str):
                return key, value['@xlink:href']

            if key == 'address':
                return key, value['addr-line']

            # Single occurence
            path_list = (
                'front/article-meta/abstract',
                'front/article-meta/author-notes',
                'front/article-meta/bio',
            )
            if xpath in set(path_list) and not key.startswith('@'):
                element = article_et.xpath(xpath + '[not(@processed="true")]')[0]
                value = etree_str(element).strip()
                element.set('processed', 'true')
                if not value:
                    return None
                return key, str(value)
            if re.search('|'.join(map(lambda x: '^' + x, path_list)), xpath):
                return None

            # Multiple lines per occurrence
            path_list = (
                'back/app-group/app',
                'back/fn-group',
                'back/sec',
                'front/article-meta/contrib-group/fn',
            )
            if xpath in set(path_list) and not key.startswith('@'):
                element = article_et.xpath(xpath + '[not(@processed="true")]')[0]
                value = etree_str(element).strip()
                element.set('processed', 'true')
                if not value:
                    return None
                return key, value
            if re.search('|'.join(map(lambda x: '^' + x, path_list)), xpath):
                return None

            # Multiple lines per occurrence
            if key in set((
                    'notes',
                    'bio',
                    'ack',
                    'fn',
                    'sec',
                    'app',
            )):
                element = article_et.xpath(xpath + '[not(@processed="true")]')[0]
                value = etree_str(element).strip()
                element.set('processed', 'true')
                if not value:
                    return None
                return key, value

            # Single line per occurrence
            if key in set((
                    'aff',
                    'collab',
                    'string-name',
                    'title',
                    'subtitle',
                    'label',
                    'mixed-citation',
                    'subject',
                    'kwd',
                    'addr-line',
                    'article-title',
                    'issue-id',
                    'issue',
                    'volume',
            )):
                element = article_et.xpath(xpath + '[not(@processed="true")]')[0]
                value = etree_str(element).strip()
                element.set('processed', 'true')
                if not value:
                    return None
                return key, re.sub('\s+', ' ', str(value))

            return key, value

        return postprocessor

    def parse(self, xml_path):
        article_et = etree.parse(xml_path, parser=etree.XMLParser(recover=True)).getroot()
        front = article_et.xpath('front')[0]
        front_back = front.xpath('back')
        front_body = front.xpath('body')
        if front_back:
            back = front_back[0]
            front.remove(back)
            article_et.append(str(back))
        if front_body:
            body = front_body[0]
            front.remove(body)
            article_et.append(str(body))

        if not front_body and not front_back:
            with open(xml_path, 'r') as fh:
                article = xmltodict.parse(fh.read(),
                                          postprocessor=self._get_parse_postprocessor(article_et),
                                          force_list=('string-name', 'contrib-group', 'contrib', 'aff'),
                                          )['article']
        else:
            article = \
                xmltodict.parse(etree.tostring(article_et), postprocessor=self._get_parse_postprocessor(article_et))[
                    'article']

        article['front']['journal-meta'] = self._process__journal_meta(article['front']['journal-meta'])
        article['front']['article-meta'] = self._process__article_meta(article['front']['article-meta'])

        self._process_if_avail(article['front'], 'notes', lambda v: v if isinstance(v, list) else [v])
        self._process_if_avail(article.get('back', {}), 'sec', lambda v: v if isinstance(v, list) else [v])

        article.update(article.pop('front'))

        try:
            article.update(article.pop('back'))
        except KeyError:
            pass

        return article

class TXTParser(object):
    def parse(self, txt_path):
        txt_root = ET.parse(txt_path).getroot()
        print(txt_root.tag)
        if txt_root.tag == 'plain_text':
            page_seq = [(p.attrib['sequence'], p.text) for p in list(txt_root)]
            page_seq.sort(key=lambda x: x[0])
            plain_text_pages = [p for s, p in page_seq]
            return {'plain_text': str(plain_text_pages)}
        elif txt_root.tag == 'body':
            return {'body': str(ET.tostring(txt_root, encoding='utf-8', method='text'))}


class EmbeddingParser(object):

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

    def word_to_vec(self, text, word_type):
        marked_text = "[CLS] " + text + " [SEP]"

        # Split the sentence into tokens.
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        #logger.debug("Sentence: {}".format(marked_text))
        #logger.debug("Length of tokens_tensor {} and segments_tensor {} ".format(
             #str(len(indexed_tokens)), str(len(segments_ids))))

        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        token_vecs_sum = []
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)

        #change later to handle exception
        target_index = 0

        for i, token_str in enumerate(tokenized_text):
            if token_str == word_type.value:
                target_index = i
                break
        return token_vecs_sum[target_index].numpy().tolist()

    def parse(self, type, pages):
        result = []
        if type == EmbeddingType.SPECIES:
            for page in pages:
                sentences = split_into_sentences(page)
                for sentence in sentences:
                    # Truncating exception sentences which are of unusually large size)
                    if(len(sentence) > 700):
                        t_sentence = "TRUNC: " + sentence[:700]
                    else:
                        t_sentence = sentence

                    if self.contains_keyword(t_sentence, EmbeddingType.SPECIES):
                        result_dict = {}
                        result_dict['sentence'] = t_sentence
                        result_dict['embedding'] = self.word_to_vec(t_sentence, EmbeddingType.SPECIES)
                        result.append(result_dict)
            return {'bert_embeddings_species': result}
        elif type == EmbeddingType.SUBSPECIES:
            for page in pages:
                sentences = split_into_sentences(page)
                for sentence in sentences:
                    # Truncating exception sentences which are of unusually large size)
                    if(len(sentence) > 700):
                        t_sentence = "TRUNC: " + sentence[:700]
                    else:
                        t_sentence = sentence

                    if self.contains_keyword(t_sentence, EmbeddingType.SUBSPECIES):
                        result_dict = {}
                        result_dict['sentence'] = t_sentence
                        result_dict['embedding'] = self.word_to_vec(t_sentence, EmbeddingType.SUBSPECIES)
                        result.append(result_dict)
            return {'bert_embeddings_subspecies': result}
        else:
            return {'bert_embeddings': []}

    def contains_keyword(self, sentence, SPECIES):
        if SPECIES.value in sentence.split():
            return True
        else:
            return False


def generate_actions(dataset_dir, index, document_type):
    abs_dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    xmlparser = XMLParser()
    txtparser = TXTParser()
    embeddingparser = EmbeddingParser()
    # doc_id = es.count(index, document_type)['count']

    if DEBUG:
        abs_dataset_dir = DEBUG
        import pdb
        pdb.set_trace()

    for (dpath, dnames, fnames) in os.walk(abs_dataset_dir):
        if dnames:
            dnames.sort()

        if not fnames:
            continue

        logger.debug('Processing %s' % dpath)
        document = {}

        xml_files = [p for p in fnames if p.lower().endswith('.xml')]
        txt_files = [p for p in fnames if p.lower().endswith('.txt')]

        if len(xml_files) == 1 and len(txt_files) == 1:
            xml_path = os.path.join(dpath, xml_files[0])
            txt_path = os.path.join(dpath, txt_files[0])

            try:
                document['article'] = xmlparser.parse(os.path.join(dpath, xml_path))
            except LanguageError as e:
                logger.warning('{}: language \'{}\' not en/eng'.format(xml_path, e))
                continue
            except Exception as e:
                logger.error('{}: {}'.format(xml_path, e))
                print(traceback.format_exc())
                continue

            try:
                txt = txtparser.parse(txt_path)
                document.update(txt)
                document.update(embeddingparser.parse(EmbeddingType.SPECIES, list(txt.values())))
                document.update(embeddingparser.parse(EmbeddingType.SUBSPECIES, list(txt.values())))
            except ET.ParseError as e:
                logger.error('{}: {}'.format(txt_path, e))
                continue

            action = {
                '_index': index,
                '_type': document_type,
                # '_id': doc_id,
                '_source': document
            }
            # doc_id += 1
            yield action


action_generator = generate_actions(DATASET_DIR, index=ES_INDEX_NAME, document_type=ES_DOCUMENT_TYPE)

#ELASTICSEARCH UPLOAD
es = elasticsearch.Elasticsearch([ES_HOST], timeout=1000, max_retries=5, retry_on_timeout=True)

elasticsearch.helpers.bulk(es, action_generator, chunk_size=100, timeout="1000s")

#VERIFY COUNT
print(es.count(index=ES_INDEX_NAME)['count'])
