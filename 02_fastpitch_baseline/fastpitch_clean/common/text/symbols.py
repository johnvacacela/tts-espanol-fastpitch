""" from https://github.com/keithito/tacotron """
from .cmudict import valid_symbols

_arpabet = ["@" + s for s in valid_symbols]


def get_symbols(symbol_set="english_basic"):
    if symbol_set == "english_basic":
        _pad = "_"
        _punctuation = "!\'(),.:;? "
        _special = "-"
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == "english_basic_lowercase":
        _pad = "_"
        _punctuation = "!\'\",.:;? "
        _special = "-"
        _letters = "abcdefghijklmnopqrstuvwxyz"
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == "english_expanded":
        _punctuation = "!\'\",.:;? "
        _math = "#%&*+-/[]()"
        _special = "_@\u00a9\u00b0\u00bd\u2014\u20a9\u20ac$"
        _accented = "\u00e1\u00e7\u00e9\u00ea\u00eb\u00f1\u00f6\u00f8\u0107\u017e"
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        symbols = list(_punctuation + _math + _special + _accented + _letters) + _arpabet
    elif symbol_set == "ipa_all":
        symbols = [
            "_",
            "a", "b", "c", "d\u032a", "e", "f", "i", "j", "k", "l", "m",
            "n", "o", "p", "r", "s", "t\u0283", "t\u032a", "u", "w", "x",
            "\u00f0", "\u014b", "\u025f", "\u025f\u029d", "\u0261", "\u0263",
            "\u0272", "\u027e", "\u0283", "\u028e", "\u029d", "\u03b2", "\u03b8",
            "sil"
        ]
    else:
        raise Exception("{} symbol set does not exist".format(symbol_set))
    return symbols


def get_pad_idx(symbol_set="english_basic"):
    if symbol_set in {"english_basic", "english_basic_lowercase", "ipa_all"}:
        return 0
    else:
        raise Exception("{} symbol set not used yet".format(symbol_set))
