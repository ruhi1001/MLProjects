# ---------------------------------
# Created By : ruhi.ahuja
# Created Date : 
# ---------------------------------

"""
*** File details here
"""
import charset_normalizer
import pandas as pd


class CharacterEncoding:
    def main_method(self):
        before = 'This is the euro symbol: €'
        print(type(before))

        # The other data is the bytes data type, which is a sequence of integers.
        # You can convert a string into bytes by specifying which encoding it's in:

        after = before.encode("utf-8", errors="replace")
        print(type(after))
        print(after)
        # convert it back to utf-8
        print(after.decode("utf-8"))
        # try to decode our bytes with the ascii encoding
        # print(after.decode("ascii"))

        # strings are UTF-8 by default in Python 3, so if we try to treat them like they were
        # in another encoding we'll create problems.

        # Reading in files with encoding problems:
        # look at the first ten thousand bytes to guess the character encoding:
        with open('ks-projects-201612.csv', 'rb') as rawdata:
            result = charset_normalizer.detect(rawdata.read(10000))

        print(result)
        kickstarter_2016 = pd.read_csv('ks-projects-201612.csv', encoding='ISO-8859-1')
        print(kickstarter_2016.head())


if __name__=='__main__':
    CharacterEncoding().main_method()