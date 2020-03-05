import logging
logger = logging.getLogger()

import editdistance


def cleanTermList(termList):
    '''Remove items from a list, which are 'too close' to other items on the
    list. In this way, only one version is retained. Closeness is measured as
    the edit distance between any two words, normalized by the length of
    the shortest word.

    termList    A list of (word,weight) tuples to be filtered.
    '''
    logger.debug('enter %s.cleanTermList', __name__)
    minEditDiff = 0.20
    cleanTerms = []
    for word, weight in termList:
        if not _isCloseToList(word, cleanTerms, minEditDiff):
            cleanTerms.append((word, weight))
    logger.debug('exit %s.cleanTermList', __name__)
    return cleanTerms


def _isCloseToList(word, cleanTerms, minEditDiff):
    logger.debug('enter %s._isCloseToList', __name__)
    if len(cleanTerms) == 0:
        logger.debug('exit %s._isCloseToList', __name__)
        return False
    for known, _ in cleanTerms:
        diff = float(editdistance.eval(word, known)) / \
            min(len(word), len(known))
        if diff < minEditDiff:
            # print '%s is too close (%2.4f) to %s'%(word, diff, known)
            logger.debug('exit %s._isCloseToList', __name__)
            return True
    logger.debug('exit %s._isCloseToList', __name__)
    return False
