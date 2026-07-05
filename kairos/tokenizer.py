from transformers.models.byt5.tokenization_byt5 import ByT5Tokenizer


class KairosTokenizer(ByT5Tokenizer):
    pass



# ==========================================================
# Reserved Special Tokens
# ==========================================================
#
# 0-255 : Raw byte values
#
# 256 : <PAD>
# 257 : <BOS>
# 258 : <EOS>
#
# -------- Modalities --------
#
# 259 : <TEXT>
# 260 : </TEXT>
#
# 261 : <IMG>
# 262 : </IMG>
#
# 263 : <VIDEO>
# 264 : </VIDEO>
#
# 265 : <AUDIO>
# 266 : </AUDIO>
#
# 267 : <LIDAR>
# 268 : </LIDAR>
#
# 269 : <STATE>
# 270 : </STATE>
#
# 271 : <ACTION>
# 272 : </ACTION>
#
# -------- Image Channels --------
#
# 273 : <R>
# 274 : </R>
#
# 275 : <G>
# 276 : </G>
#
# 277 : <B>
# 278 : </B>
#
# -------- Audio Channels --------
#
# 279 : <LEFT>
# 280 : </LEFT>
#
# 281 : <RIGHT>
# 282 : </RIGHT>
#
# -------- Optional Metadata --------
#
# 283 : <META>
# 284 : </META>
#
# 285 : <TIMESTAMP>
# 286 : </TIMESTAMP>
#
# 287 : <SEP>
# 288 : <MASK>
#
# ==========================================================
# Total vocab size = 289
# ==========================================================
