import unittest
from sortedcontainers import SortedDict, SortedList
from shico import VocabularyAggregator as shVA
from shico.vocabularyaggregator import _arrangeIntervals

class TestVocabularyAggregation(unittest.TestCase):
    '''Tests for VocabularyAggregator'''
    @classmethod
    def setUpClass(self):
        self._data = SortedDict({
            '1950_1959': [('bevrijding', 1), ('wereldoorlog', 1),
                          ('oorlogen', 1), ('burgeroorlog', 1),
                          ('oorlof', 1), ('bevr\xfcding', 1), ('ooriog', 1),
                          ('hongerwinter', 1), ('oorlogse', 1), ('oorlog', 1)
                          ],
            '1951_1960': [('oorlog', 8), ('bevrijding', 4),
                          ('souvereiniteitsoverdracht', 3), ('oorloe', 3),
                          ('ooriogse', 3), ('bevr\xfcding', 3),
                          ('ooriog', 3), ('oorlogsjaren', 3), ('corlog', 2),
                          ('boerenoorlog', 2)
                          ],
            '1952_1961': [('oorlog', 5), ('bevrijding', 4),
                          ('geboorteplek', 4), ('bevryding', 4),
                          ('herstelperiode', 3), ('bezettingsjaren', 3),
                          ('oorlogse', 3), ('crisisperiode', 3),
                          ('bevr\xfcding', 3), ('oorlogsjaren', 3)
                          ],
            '1953_1962': [('oorlog', 6), ('bevrijding', 5),
                          ('geboorteplek', 4), ('bevr\xfcding', 4),
                          ('souvereiniteitsoverdracht', 3),
                          ('bevrijdingsoorlog', 3), ('bezettingsjaren', 3),
                          ('wereldoorlog', 3), ('oorlogse', 3),
                          ('bevryding', 3)
                          ],
        })

    def testWeightingFunctions(self):
        '''Test that VocabularyAggregator supports weighting functions and fails
        for unsupported ones.'''
        for f in ['Gaussian', 'JSD',  'Linear']:
            try:
                agg = shVA(weighF=f)
                agg.aggregate(self._data)
            except:
                self.fail(f + ' should be a valid function')

        try:
            agg = shVA(weighF=lambda t1, t2: 0)
            agg.aggregate(self._data)
        except:
            self.fail('Lambda function should be a valid function')

        with self.assertRaises(Exception):
            agg = shVA(weighF='Unknown')
            agg.aggregate(self._data)

    def testWordsPerYear(self):
        '''Test that aggregator produces the correct number of results'''
        nWordsPerYear = 5
        agg = shVA(nWordsPerYear=nWordsPerYear)
        aggData, _ = agg.aggregate(self._data)
        for words in aggData.values():
            self.assertEqual(len(words), nWordsPerYear,
                             'Each year should have %d words ' % nWordsPerYear)

    def testYearsInInterval(self):
        '''Test aggregator reduces the number of intervals produced when
        such intervals are longer'''
        agg = shVA(yearsInInterval=1)
        aggData, _ = agg.aggregate(self._data)
        self.assertEqual(len(list(aggData.keys())), len(self._data),
                         'Should have same number of keys as original data')

        agg = shVA(yearsInInterval=2)
        aggData, _ = agg.aggregate(self._data)
        self.assertEqual(len(list(aggData.keys())), len(self._data)/2,
                         'Should have 1/2 the number of keys as original data')

        agg = shVA(yearsInInterval=len(self._data))
        aggData, _ = agg.aggregate(self._data)
        self.assertEqual(len(list(aggData.keys())), 1,
                         'Should have only 1 key')

        agg = shVA(yearsInInterval=2 * len(self._data))
        aggData, _ = agg.aggregate(self._data)
        self.assertEqual(len(list(aggData.keys())), 1,
                         'Should have only 1 key, containing all years')

    def testTimePeriods(self):
        '''Test aggregator produces metadata'''
        agg = shVA(yearsInInterval=1, yIntervalFreq=1)
        data, times = agg.aggregate(self._data)
        self.assertEqual(len(data), len(times),
                         'Should have same number of keys')
        self.assertTrue(list(data.keys()) == list(times.keys()),
                        'Should be the same keys')

        yearsInInterval = 2
        agg = shVA(yearsInInterval=yearsInInterval, yIntervalFreq=1)
        _, times = agg.aggregate(self._data)
        for year, values in times.items():
            self.assertEqual(len(values), yearsInInterval,
                             'Should have equal number of years in interval '
                             'but %s does not' % year)

        agg1 = shVA(yearsInInterval=yearsInInterval, yIntervalFreq=1)
        agg2 = shVA(yearsInInterval=yearsInInterval, yIntervalFreq=2)
        _, times1 = agg1.aggregate(self._data)
        _, times2 = agg2.aggregate(self._data)
        self.assertGreater(len(times1), len(times2),
                           'Should have more intervals')

    def testArrangeIntervals1(self):
        targetKeys = SortedList(['1950_1959', '1951_1960', '1952_1961', '1953_1962', '1954_1963'])
        targetIntervals = [ ['1950_1959', '1951_1960', '1952_1961'],
                            ['1951_1960', '1952_1961', '1953_1962'],
                            ['1952_1961', '1953_1962', '1954_1963'] ]
        seedVocabulary = SortedDict({
            key: {} for key in targetKeys
        })

        actualKeys = SortedList(list(seedVocabulary.keys()))
        actualIntervals = _arrangeIntervals(seedVocabulary, 3, 1)
        self._doArrangeIntervalsTesting(targetKeys, actualKeys, targetIntervals, actualIntervals, 'Test1')

    def testArrangeIntervals2(self):
        targetKeys = SortedList(['1950_1959', '1952_1961', '1954_1963'])
        targetIntervals = [ ['1950_1959', '1952_1961'],
                            ['1952_1961', '1954_1963'] ]
        seedVocabulary = SortedDict({
            key: {} for key in targetKeys
        })

        actualKeys = SortedList(list(seedVocabulary.keys()))
        actualIntervals = _arrangeIntervals(seedVocabulary, 2, 1)
        self._doArrangeIntervalsTesting(targetKeys, actualKeys, targetIntervals, actualIntervals, 'Test1')


    def testArrangeIntervals3(self):
        targetKeys = SortedList(
        ['1835_1855', '1840_1860', '1845_1865', '1850_1870', '1855_1875',
         '1860_1880', '1865_1885', '1870_1890', '1875_1895'])
        targetIntervals = [ ['1835_1855', '1840_1860', '1845_1865'],
                            ['1840_1860', '1845_1865', '1850_1870'],
                            ['1845_1865', '1850_1870', '1855_1875'],
                            ['1850_1870', '1855_1875', '1860_1880'],
                            ['1855_1875', '1860_1880', '1865_1885'],
                            ['1860_1880', '1865_1885', '1870_1890'],
                            ['1865_1885', '1870_1890', '1875_1895'] ]
        seedVocabulary = SortedDict({
            key: {} for key in targetKeys
        })

        actualKeys = SortedList(list(seedVocabulary.keys()))
        actualIntervals = _arrangeIntervals(seedVocabulary, 3, 1)
        self._doArrangeIntervalsTesting(targetKeys, actualKeys, targetIntervals, actualIntervals, 'Test2')

    def _doArrangeIntervalsTesting(self, targetKeys, actualKeys, targetIntervals, actualIntervals, testname):
        self.assertSequenceEqual(targetKeys, actualKeys, 'Should have same keys.')
        self.assertSequenceEqual(targetIntervals, actualIntervals, 'Should have same intervals.')
