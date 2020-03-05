import logging
logger = logging.getLogger()

from collections import Counter
import numpy as np


def getRangeMiddle(first, last=None):
    '''Find the year in the middle of a year range. The range can be given
    either by a single string in the format YEAR1_YEAR2, or by two strings
    in this format.'''
    logger.debug('enter %s.getRangeMiddle', __name__)
    if last is None:
        last = first
    y0 = int(first.split('_')[0])
    yn = int(last.split('_')[1])
    logger.debug('exit %s.getRangeMiddle', __name__)
    return round((yn + y0) / 2)


def yearlyNetwork(aggPeriods, aggResults, results, links):
    '''Build a dictionary of network graph definitions. The key of this
    dictionary are the years and the values are the network definition
    (in the format used by D3).'''
    logger.debug('enter %s.yearlyNetwork', __name__)
    seeds = {y: list(seedResp.keys()) for y, seedResp in links.items()}

    networks = {}
    for year_mu, years in aggPeriods.items():
        yResults = {y: results[y] for y in years}
        yLinks = {y: links[y] for y in years}
        ySeeds = {y: seeds[y] for y in years}

        finalVocab = aggResults[year_mu]
        networks[year_mu] = _metaToNetwork(
            yResults, ySeeds, finalVocab, yLinks)
    logger.debug('exit %s.yearlyNetwork', __name__)
    return networks


def yearTuplesAsDict(results):
    '''Converts the values of a dictionary from tuples to dictionaries.
    E.g.
    From:
        {
            1950: ('a',w1), ('b',w2),('c',w3)
        }
    To:
        {
            1950: {
                'a': w1,
                'b': w2,
                'c': w3
            }
        }
    '''
    logger.debug('touch %s.yearTuplesAsDict', __name__)
    return {year: _tuplesAsDict(vals) for year, vals in results.items()}


def _buildNode(word, counts, seedSet, finalWords):
    ''' Build a node for a force directed graph in format used by front end'''
    logger.debug('enter %s._buildNode', __name__)
    if word in seedSet:
        nodeType = 'seed'
    elif word in finalWords:
        nodeType = 'word'
    else:
        nodeType = 'drop'
    logger.debug('exit %s._buildNode', __name__)
    return {
        'name': word,
        'count': counts[word],
        'type': nodeType
    }


def _buildLink(seed, word, distance, nodeIdx):
    ''' Build a node for a force directed graph in format used by front end'''
    logger.debug('enter %s._buildLink', __name__)
    seedIdx = nodeIdx[seed]
    wordIdx = nodeIdx[word]
    logger.debug('exit %s._buildLink', __name__)
    return {
        'source': seedIdx,
        'target': wordIdx,
        'value':  1 / (distance + 1)
    }


def _buildLinks(yLinks, nodeIdx):
    '''Build a list of links from the given yearly links. For each year group,
    link all seeds to all their results (with strength proportional to their
    distance)'''
    logger.debug('enter %s._buildLinks', __name__)
    linkList = []
    # We are not doing anything with the years in yLinks
    for links in list(yLinks.values()):
        for seed, results in links.items():
            for word, distance in results:
                # TODO: check seeds present in dict more elegantly
                if seed in nodeIdx and word in nodeIdx:
                    linkList.append(_buildLink(seed, word, distance, nodeIdx))
                else:
                    print('Seed or word not in index!')
    logger.debug('exit %s._buildLinks', __name__)
    return linkList


def _buildNodes(wordCounts, seedSet, finalWords):
    '''Build a set of nodes for the network graph. Also builds an dictionary of
    words and the ID's used to represent them on the list of nodes.'''
    logger.debug('enter %s._buildNodes', __name__)
    nodeIdx = {}
    nodes = []
    # Make nodes from unique words
    uniqueWords = set(list(wordCounts.keys()) + list(seedSet))
    for idx, w in enumerate(uniqueWords):
        nodes.append(_buildNode(w, wordCounts, seedSet, finalWords))
        nodeIdx[w] = idx
    logger.debug('exit %s._buildNodes', __name__)
    return nodes, nodeIdx


def _metaToNetwork(results, seeds, finalVocab, yLinks):
    '''Build a network (nodes & links), using aggregated results. Network must
    be in a format usable by the front end. '''
    logger.debug('enter %s._metaToNetwork', __name__)
    wordCounts = Counter([w for res in list(results.values()) for w, v in res])
    seedSet = set(w for seed in list(seeds.values()) for w in seed)
    finalWords = [w for w, v in finalVocab]

    nodes, nodeIdx = _buildNodes(wordCounts, seedSet, finalWords)
    links = _buildLinks(yLinks, nodeIdx)

    network = {
        'nodes': nodes,
        'links': links
    }
    logger.debug('exit %s._metaToNetwork', __name__)
    return network


def _tuplesAsDict(pairList):
    '''Convert list of (words,weight) to dict of word: weight'''
    logger.debug('touch %s._tuplesAsDict', __name__)
    return {word: weight for word, weight in pairList}


def wordLocationAsDict(word,loc):
    '''Wrap the given word and it's (x,y) location in a dictionary.'''
    logger.debug('touch %s.wordLocationAsDict', __name__)
    return {
        'word': word,
        'x': 0 if np.isnan(loc[0]) else loc[0],
        'y': 0 if np.isnan(loc[1]) else loc[1]
    }
