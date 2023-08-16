    #!/usr/bin/env python
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from collections import defaultdict
import numpy as np
import math
import pickle
from lib.config import cfg

def precook(words, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    #words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n, True)

class CiderScorer(object):
    """CIDEr scorer.
    """
    def copy(self):
        ''' copy the refs.'''
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        new.cache_scores = self.cache_scores
        new.cache_vecs = self.cache_vecs
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []

        cider_cache = pickle.load(open(cfg.SCORER.CIDER_CACHED, 'rb'), encoding='bytes')
        self.document_frequency = cider_cache['document_frequency']
        self.ref_len = cider_cache['ref_len']
        self.cook_append(test, refs)
        self.clear_cache()
        
        
    def clear(self):
        self.crefs = []
        self.ctest = []
        #self.cache_scores = {}

#     def cook_append(self, test, refs):
#         '''called by constructor and __iadd__ to avoid creating new instances.'''

#         if refs is not None:
#             self.crefs.append(cook_refs(refs))
#             if test is not None:
#                 self.ctest.append(cook_test(test)) ## N.B.: -1
#             else:
#                 self.ctest.append(None) # lens of crefs and ctest have to match
                
    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.crefs.append(refs)
            if test is not None:
                self.ctest.append(test) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            #print("cook_append")
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self

    def compute_doc_freq(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    
    def clear_cache(self):
        self.cache_scores = {}
        self.cache_vecs = {}
        self.cache_count = {}
            
    def compute_cider(self):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val
        
        def process_vecs_with_cache(id_list, id_str, cache_obj):
            if (cache_obj.get(id_str, None) is not None):
                vec, norm, length = cache_obj.get(id_str)
                #print("cache_vec", vec, norm, length)
            else:
                vec, norm, length = counts2vec(precook(id_list))
                cache_obj[id_str] = (vec, norm, length)
            return vec, norm, length

        # compute log reference length
        # self.ref_len = np.log(float(len(self.crefs))) ###########################
        
        ctest_str = [' '.join([str(ind) for ind in tempList]) for tempList in self.ctest]
        crefs_str = [[' '.join([str(ind) for ind in tempList]) for tempList in temp_crefs] for temp_crefs in self.crefs]
        
        scores = []
        
        for i, (test, refs) in enumerate(zip(self.ctest, self.crefs)):
            # compute vector for test captions
            
            vec, norm, length = process_vecs_with_cache(test, ctest_str[i], self.cache_vecs)
                
            #vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for j, ref in enumerate(refs):
                
                vec_ref, norm_ref, length_ref = process_vecs_with_cache(ref, crefs_str[i][j], self.cache_vecs)
                #vec_ref, norm_ref, length_ref = counts2vec(ref)
                
                combine1 = ctest_str[i] + ' ' + crefs_str[i][j]
                combine2 = crefs_str[i][j] + ' ' + ctest_str[i]

                if (self.cache_scores.get(combine1, None) is not None):
                    tempscore = self.cache_scores[combine1]
                    #print("combine1", combine1, tempscore)
                elif (self.cache_scores.get(combine2, None) is not None):
                    tempscore = self.cache_scores[combine2]
                    #print("combine2", combine2, tempscore)
                else:
                    tempscore = sim(vec, vec_ref, norm, norm_ref, length, length_ref)
                    
                    self.cache_scores[combine1] = tempscore
                    self.cache_scores[combine2] = tempscore 
                
                    
                #add for each ref
                score += tempscore
                
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        # compute cider score
        score = self.compute_cider()
        # debug
        # print score
        return np.mean(np.array(score)), np.array(score)