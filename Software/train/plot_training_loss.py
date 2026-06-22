"""
Plot training and validation loss curves from training log output.

Usage:
    python plot_training_loss.py <log_file>
    python plot_training_loss.py training_log.txt

Or paste the log directly into TRAINING_LOG below and run without args.
"""

import re
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Paste your training log here (or load from file via --log argument)
TRAINING_LOG = r"""
[E000] train=1.016982  val=0.988713  lr=1.42e-05  time=265.2s
[E001] train=0.649815  val=0.612728  lr=2.84e-05  time=224.0s
[E002] train=0.238938  val=0.273592  lr=4.26e-05  time=257.5s
[E003] train=0.147274  val=0.161563  lr=5.68e-05  time=296.4s
[E004] train=0.116973  val=0.114521  lr=7.10e-05  time=296.8s
[E005] train=0.094387  val=0.093018  lr=8.52e-05  time=297.0s
[E006] train=0.084918  val=0.083046  lr=9.94e-05  time=297.2s
[E007] train=0.072869  val=0.069785  lr=1.00e-04  time=300.2s
[E008] train=0.065408  val=0.065822  lr=1.00e-04  time=297.7s
[E009] train=0.059227  val=0.058533  lr=1.00e-04  time=298.1s
[E010] train=0.051992  val=0.056608  lr=1.00e-04  time=296.2s
[E011] train=0.050274  val=0.049559  lr=1.00e-04  time=298.8s
[E012] train=0.045534  val=0.050505  lr=1.00e-04  time=298.4s
[E013] train=0.044060  val=0.045552  lr=1.00e-04  time=299.0s
[E014] train=0.043170  val=0.042545  lr=1.00e-04  time=298.0s
[E015] train=0.038311  val=0.045308  lr=1.00e-04  time=297.3s
[E016] train=0.037156  val=0.040231  lr=1.00e-04  time=298.4s
[E017] train=0.037877  val=0.040553  lr=1.00e-04  time=299.6s
[E018] train=0.037133  val=0.041940  lr=1.00e-04  time=298.8s
[E019] train=0.036811  val=0.042006  lr=1.00e-04  time=298.9s
[E020] train=0.037172  val=0.037751  lr=1.00e-04  time=299.6s
[E021] train=0.035747  val=0.038775  lr=1.00e-04  time=298.0s
[E022] train=0.037350  val=0.035711  lr=1.00e-04  time=298.1s
[E023] train=0.034841  val=0.034801  lr=1.00e-04  time=297.1s
[E024] train=0.031250  val=0.036618  lr=1.00e-04  time=297.6s
[E025] train=0.032238  val=0.036756  lr=1.00e-04  time=297.3s
[E026] train=0.034431  val=0.040899  lr=1.00e-04  time=296.8s
[E027] train=0.031529  val=0.034751  lr=1.00e-04  time=296.5s
[E028] train=0.033106  val=0.032432  lr=1.00e-04  time=296.4s
[E029] train=0.031693  val=0.032275  lr=1.00e-04  time=297.5s
[E030] train=0.029141  val=0.039158  lr=1.00e-04  time=299.0s
[E031] train=0.028735  val=0.033105  lr=1.00e-04  time=298.0s
[E032] train=0.028671  val=0.034156  lr=1.00e-04  time=298.4s
[E033] train=0.026570  val=0.035664  lr=1.00e-04  time=298.1s
[E034] train=0.027540  val=0.031124  lr=1.00e-04  time=298.5s
[E035] train=0.025028  val=0.034301  lr=1.00e-04  time=298.4s
[E036] train=0.026681  val=0.030133  lr=1.00e-04  time=297.5s
[E037] train=0.030306  val=0.029070  lr=1.00e-04  time=298.2s
[E038] train=0.029127  val=0.032965  lr=1.00e-04  time=298.0s
[E039] train=0.027622  val=0.038374  lr=1.00e-04  time=298.0s
[E040] train=0.027427  val=0.030716  lr=1.00e-04  time=297.3s
[E041] train=0.027035  val=0.030694  lr=1.00e-04  time=298.1s
[E042] train=0.023838  val=0.030721  lr=1.00e-04  time=298.7s
[E043] train=0.024576  val=0.031748  lr=1.00e-04  time=299.1s
[E044] train=0.029141  val=0.029186  lr=1.00e-04  time=299.0s
[E045] train=0.028539  val=0.026837  lr=1.00e-04  time=298.8s
[E046] train=0.026451  val=0.030763  lr=1.00e-04  time=299.4s
[E047] train=0.025367  val=0.027617  lr=1.00e-04  time=298.4s
[E048] train=0.026324  val=0.025553  lr=1.00e-04  time=298.3s
[E049] train=0.026796  val=0.029242  lr=1.00e-04  time=299.7s
[E050] train=0.023347  val=0.025756  lr=1.00e-04  time=294.7s
[E051] train=0.025001  val=0.030386  lr=1.00e-04  time=298.7s
[E052] train=0.025936  val=0.030079  lr=1.00e-04  time=298.2s
[E053] train=0.026241  val=0.025481  lr=1.00e-04  time=297.8s
[E054] train=0.025923  val=0.026116  lr=1.00e-04  time=298.2s
[E055] train=0.026661  val=0.023033  lr=1.00e-04  time=296.8s
[E056] train=0.023273  val=0.030247  lr=1.00e-04  time=298.4s
[E057] train=0.022640  val=0.031125  lr=1.00e-04  time=297.7s
[E058] train=0.023629  val=0.027289  lr=1.00e-04  time=296.9s
[E059] train=0.023913  val=0.028760  lr=1.00e-04  time=297.8s
[E060] train=0.022534  val=0.025495  lr=1.00e-04  time=296.9s
[E061] train=0.023134  val=0.025198  lr=1.00e-04  time=297.4s
[E062] train=0.024380  val=0.027522  lr=1.00e-04  time=297.0s
[E063] train=0.025406  val=0.024621  lr=1.00e-04  time=297.7s
[E064] train=0.024897  val=0.027038  lr=1.00e-04  time=296.6s
[E065] train=0.024165  val=0.026256  lr=1.00e-04  time=298.2s
[E066] train=0.024884  val=0.025896  lr=1.00e-04  time=297.6s
[E067] train=0.021891  val=0.024358  lr=1.00e-04  time=297.8s
[E068] train=0.021441  val=0.025470  lr=1.00e-04  time=298.4s
[E069] train=0.024627  val=0.030393  lr=1.00e-04  time=298.5s
[E070] train=0.023314  val=0.027249  lr=1.00e-04  time=297.6s
[E071] train=0.022098  val=0.025951  lr=1.00e-04  time=299.1s
[E072] train=0.024444  val=0.023696  lr=1.00e-04  time=299.2s
[E073] train=0.022524  val=0.026711  lr=1.00e-04  time=298.7s
[E074] train=0.020132  val=0.025376  lr=1.00e-04  time=300.0s
[E075] train=0.021785  val=0.028451  lr=1.00e-04  time=300.1s
[E076] train=0.022551  val=0.024809  lr=1.00e-04  time=298.7s
[E077] train=0.023647  val=0.026356  lr=1.00e-04  time=299.6s
[E078] train=0.020085  val=0.026171  lr=1.00e-04  time=297.8s
[E079] train=0.021558  val=0.024519  lr=1.00e-04  time=298.0s
[E080] train=0.020343  val=0.024967  lr=1.00e-04  time=298.8s
[E081] train=0.020791  val=0.027102  lr=1.00e-04  time=297.0s
[E082] train=0.023244  val=0.027631  lr=1.00e-04  time=295.2s
[E083] train=0.021908  val=0.024809  lr=1.00e-04  time=296.4s
[E084] train=0.020270  val=0.022616  lr=1.00e-04  time=296.7s
[E085] train=0.023521  val=0.023861  lr=1.00e-04  time=296.2s
[E086] train=0.021587  val=0.025381  lr=1.00e-04  time=298.0s
[E087] train=0.020980  val=0.024993  lr=1.00e-04  time=297.6s
[E088] train=0.018777  val=0.020568  lr=1.00e-04  time=296.9s
[E089] train=0.019806  val=0.024983  lr=1.00e-04  time=296.9s
[E090] train=0.020662  val=0.029263  lr=1.00e-04  time=297.5s
[E091] train=0.019688  val=0.021071  lr=1.00e-04  time=296.5s
[E092] train=0.020443  val=0.025526  lr=1.00e-04  time=296.4s
[E093] train=0.020233  val=0.024459  lr=1.00e-04  time=296.7s
[E094] train=0.018001  val=0.021910  lr=1.00e-04  time=296.8s
[E095] train=0.018861  val=0.021893  lr=1.00e-04  time=297.2s
[E096] train=0.018987  val=0.026705  lr=1.00e-04  time=297.2s
[E097] train=0.021659  val=0.025057  lr=1.00e-04  time=298.4s
[E098] train=0.019967  val=0.026512  lr=1.00e-04  time=297.8s
[E099] train=0.018880  val=0.020005  lr=1.00e-04  time=297.8s
[E100] train=0.018915  val=0.024170  lr=1.00e-04  time=298.1s
[E101] train=0.017978  val=0.021886  lr=1.00e-04  time=297.9s
[E102] train=0.022711  val=0.021847  lr=1.00e-04  time=298.6s
[E103] train=0.017794  val=0.020130  lr=1.00e-04  time=297.6s
[E104] train=0.017126  val=0.024664  lr=1.00e-04  time=298.7s
[E105] train=0.017561  val=0.024726  lr=1.00e-04  time=298.7s
[E106] train=0.019598  val=0.022489  lr=1.00e-04  time=298.8s
[E107] train=0.019445  val=0.025219  lr=1.00e-04  time=297.8s
[E108] train=0.017758  val=0.025586  lr=1.00e-04  time=298.8s
[E109] train=0.017588  val=0.023402  lr=1.00e-04  time=300.0s
[E110] train=0.016852  val=0.023610  lr=1.00e-04  time=298.3s
[E111] train=0.022363  val=0.021264  lr=1.00e-04  time=302.2s
[E112] train=0.022551  val=0.022763  lr=1.00e-04  time=307.3s
[E113] train=0.019958  val=0.024684  lr=1.00e-04  time=306.4s
[E114] train=0.018954  val=0.023797  lr=1.00e-04  time=305.9s
[E115] train=0.016845  val=0.023294  lr=1.00e-04  time=300.9s
[E116] train=0.018118  val=0.024147  lr=1.00e-04  time=297.7s
[E117] train=0.018436  val=0.021823  lr=1.00e-04  time=298.3s
[E118] train=0.015876  val=0.025298  lr=1.00e-04  time=298.8s
[E119] train=0.015785  val=0.022572  lr=1.00e-04  time=298.8s
[E120] train=0.016948  val=0.024585  lr=9.99e-05  time=298.0s
[E121] train=0.018043  val=0.023486  lr=9.99e-05  time=297.5s
[E122] train=0.017312  val=0.024306  lr=9.99e-05  time=297.7s
[E123] train=0.017363  val=0.023117  lr=9.99e-05  time=297.1s
[E124] train=0.018069  val=0.022821  lr=9.99e-05  time=297.2s
[E125] train=0.019957  val=0.020063  lr=9.99e-05  time=295.4s
[E126] train=0.017233  val=0.024412  lr=9.99e-05  time=297.0s
[E127] train=0.018631  val=0.021090  lr=9.99e-05  time=296.2s
[E128] train=0.017413  val=0.023810  lr=9.99e-05  time=296.1s
[E129] train=0.016456  val=0.020843  lr=9.99e-05  time=295.9s
[E130] train=0.019117  val=0.022757  lr=9.99e-05  time=297.9s
[E131] train=0.017664  val=0.022162  lr=9.99e-05  time=296.4s
[E132] train=0.015937  val=0.021160  lr=9.99e-05  time=297.4s
[E133] train=0.017224  val=0.024305  lr=9.99e-05  time=313.0s
[E134] train=0.016022  val=0.025097  lr=9.99e-05  time=298.1s
[E135] train=0.019283  val=0.020555  lr=9.99e-05  time=297.2s
[E136] train=0.016513  val=0.024611  lr=9.99e-05  time=298.1s
[E137] train=0.016114  val=0.024835  lr=9.99e-05  time=297.9s
[E138] train=0.016847  val=0.024603  lr=9.99e-05  time=298.2s
[E139] train=0.015678  val=0.023173  lr=9.99e-05  time=298.3s
[E140] train=0.016338  val=0.020086  lr=9.99e-05  time=299.2s
[E141] train=0.017859  val=0.019951  lr=9.99e-05  time=299.7s
[E142] train=0.018912  val=0.019898  lr=9.99e-05  time=299.9s
[E143] train=0.017711  val=0.019531  lr=9.99e-05  time=299.1s
[E144] train=0.015986  val=0.018533  lr=9.99e-05  time=299.9s
[E145] train=0.016246  val=0.023891  lr=9.99e-05  time=299.3s
[E146] train=0.015465  val=0.022015  lr=9.99e-05  time=300.2s
[E147] train=0.015340  val=0.022813  lr=9.99e-05  time=300.0s
[E148] train=0.015161  val=0.020382  lr=9.99e-05  time=299.1s
[E149] train=0.015159  val=0.019124  lr=9.99e-05  time=299.4s
[E150] train=0.015295  val=0.021011  lr=9.99e-05  time=299.0s
[E151] train=0.020760  val=0.019931  lr=9.99e-05  time=298.0s
[E152] train=0.016656  val=0.020852  lr=9.99e-05  time=297.1s
[E153] train=0.017763  val=0.021130  lr=9.99e-05  time=298.5s
[E154] train=0.014591  val=0.020980  lr=9.99e-05  time=298.0s
[E155] train=0.016813  val=0.018204  lr=9.99e-05  time=296.9s
[E156] train=0.017440  val=0.022810  lr=9.99e-05  time=296.2s
[E157] train=0.014922  val=0.020709  lr=9.99e-05  time=296.9s
[E158] train=0.014916  val=0.020745  lr=9.99e-05  time=298.0s
[E159] train=0.014785  val=0.022307  lr=9.99e-05  time=296.9s
[E160] train=0.015343  val=0.019810  lr=9.99e-05  time=296.4s
[E161] train=0.015613  val=0.021027  lr=9.99e-05  time=297.0s
[E162] train=0.017580  val=0.022668  lr=9.99e-05  time=296.5s
[E163] train=0.018052  val=0.021591  lr=9.99e-05  time=296.3s
[E164] train=0.015466  val=0.018223  lr=9.99e-05  time=296.9s
[E165] train=0.014380  val=0.019692  lr=9.99e-05  time=296.6s
[E166] train=0.013758  val=0.024360  lr=9.99e-05  time=297.4s
[E167] train=0.014413  val=0.017715  lr=9.99e-05  time=297.5s
[E168] train=0.018064  val=0.018112  lr=9.99e-05  time=297.7s
[E169] train=0.014138  val=0.022246  lr=9.99e-05  time=298.0s
[E170] train=0.013904  val=0.023465  lr=9.99e-05  time=297.7s
[E171] train=0.014745  val=0.021537  lr=9.99e-05  time=298.2s
[E172] train=0.013615  val=0.019692  lr=9.99e-05  time=297.9s
[E173] train=0.013432  val=0.021132  lr=9.99e-05  time=297.9s
[E174] train=0.013534  val=0.019585  lr=9.99e-05  time=298.4s
[E175] train=0.013972  val=0.017968  lr=9.99e-05  time=298.3s
[E176] train=0.012366  val=0.021766  lr=9.99e-05  time=299.0s
[E177] train=0.012957  val=0.017921  lr=9.99e-05  time=298.1s
[E178] train=0.016382  val=0.019613  lr=9.99e-05  time=299.1s
[E179] train=0.014759  val=0.019904  lr=9.99e-05  time=299.4s
[E180] train=0.014111  val=0.019346  lr=9.99e-05  time=299.5s
[E181] train=0.012123  val=0.021511  lr=9.99e-05  time=298.6s
[E182] train=0.014033  val=0.019743  lr=9.99e-05  time=299.2s
[E183] train=0.013561  val=0.018820  lr=9.99e-05  time=296.7s
[E184] train=0.014675  val=0.019229  lr=9.99e-05  time=298.2s
[E185] train=0.012657  val=0.019464  lr=9.99e-05  time=297.7s
[E186] train=0.015334  val=0.019103  lr=9.99e-05  time=297.5s
[E187] train=0.012993  val=0.018150  lr=9.99e-05  time=297.4s
[E188] train=0.012523  val=0.020045  lr=9.99e-05  time=297.2s
[E189] train=0.013867  val=0.019515  lr=9.99e-05  time=297.4s
[E190] train=0.012949  val=0.023366  lr=9.99e-05  time=297.1s
[E191] train=0.015271  val=0.022169  lr=9.99e-05  time=296.9s
[E192] train=0.014056  val=0.020915  lr=9.99e-05  time=296.7s
[E193] train=0.014969  val=0.019190  lr=9.99e-05  time=297.0s
[E194] train=0.016803  val=0.020355  lr=9.99e-05  time=296.3s
[E195] train=0.013336  val=0.022159  lr=9.99e-05  time=297.2s
[E196] train=0.012663  val=0.018622  lr=9.99e-05  time=296.4s
[E197] train=0.012955  val=0.021625  lr=9.99e-05  time=297.5s
[E198] train=0.015077  val=0.021304  lr=9.99e-05  time=296.5s
[E199] train=0.018426  val=0.018427  lr=9.99e-05  time=296.6s
[E200] train=0.014705  val=0.020042  lr=9.99e-05  time=298.1s
[E201] train=0.014983  val=0.016353  lr=9.99e-05  time=297.9s
[E202] train=0.016464  val=0.021061  lr=9.99e-05  time=297.8s
[E203] train=0.014052  val=0.018690  lr=9.99e-05  time=298.2s
[E204] train=0.015658  val=0.021540  lr=9.98e-05  time=298.3s
[E205] train=0.014627  val=0.020565  lr=9.98e-05  time=298.4s
[E206] train=0.014763  val=0.021128  lr=9.98e-05  time=297.1s
[E207] train=0.012022  val=0.019515  lr=9.98e-05  time=297.6s
[E208] train=0.014657  val=0.018345  lr=9.98e-05  time=298.5s
[E209] train=0.013887  val=0.017629  lr=9.98e-05  time=297.7s
[E210] train=0.013149  val=0.020579  lr=9.98e-05  time=297.4s
[E211] train=0.013815  val=0.023018  lr=9.98e-05  time=299.2s
[E212] train=0.013822  val=0.022984  lr=9.98e-05  time=298.7s
[E213] train=0.016410  val=0.015275  lr=9.98e-05  time=298.0s
[E214] train=0.013834  val=0.019270  lr=9.98e-05  time=305.3s
[E215] train=0.013220  val=0.018806  lr=9.98e-05  time=304.7s
[E216] train=0.014562  val=0.018772  lr=9.98e-05  time=298.4s
[E217] train=0.014229  val=0.020804  lr=9.98e-05  time=299.6s
[E218] train=0.012343  val=0.021786  lr=9.98e-05  time=299.8s
[E219] train=0.012640  val=0.022076  lr=9.98e-05  time=298.4s
[E220] train=0.014763  val=0.018421  lr=9.98e-05  time=298.2s
[E221] train=0.015264  val=0.020040  lr=9.98e-05  time=298.4s
[E222] train=0.013774  val=0.018917  lr=9.98e-05  time=297.0s
[E223] train=0.012394  val=0.020058  lr=9.98e-05  time=297.8s
[E224] train=0.014077  val=0.019595  lr=9.98e-05  time=297.6s
[E225] train=0.013808  val=0.020749  lr=9.98e-05  time=297.8s
[E226] train=0.014539  val=0.019962  lr=9.98e-05  time=297.1s
[E227] train=0.013329  val=0.021875  lr=9.98e-05  time=296.8s
[E228] train=0.012273  val=0.020807  lr=9.98e-05  time=297.1s
[E229] train=0.012853  val=0.018879  lr=9.98e-05  time=296.3s
[E230] train=0.012813  val=0.019035  lr=9.98e-05  time=296.8s
[E231] train=0.013041  val=0.018488  lr=9.98e-05  time=296.4s
[E232] train=0.013125  val=0.022750  lr=9.98e-05  time=299.9s
[E233] train=0.010828  val=0.016637  lr=9.98e-05  time=304.1s
[E234] train=0.013648  val=0.021580  lr=9.98e-05  time=304.3s
[E235] train=0.012529  val=0.019397  lr=9.98e-05  time=305.1s
[E236] train=0.012239  val=0.016757  lr=9.98e-05  time=296.6s
[E237] train=0.012373  val=0.016270  lr=9.98e-05  time=297.6s
[E238] train=0.011915  val=0.019233  lr=9.98e-05  time=297.0s
[E239] train=0.012521  val=0.019972  lr=9.98e-05  time=297.3s
[E240] train=0.014037  val=0.017655  lr=9.98e-05  time=306.3s
[E241] train=0.014236  val=0.018320  lr=9.98e-05  time=303.2s
[E242] train=0.011974  val=0.014567  lr=9.98e-05  time=298.2s
[E243] train=0.011621  val=0.023499  lr=9.98e-05  time=256.3s
[E244] train=0.013121  val=0.018075  lr=9.98e-05  time=223.2s
[E245] train=0.015017  val=0.016201  lr=9.98e-05  time=224.5s
[E246] train=0.013041  val=0.020676  lr=9.98e-05  time=224.6s
[E247] train=0.012506  val=0.018261  lr=9.98e-05  time=223.4s
[E248] train=0.012245  val=0.018713  lr=9.98e-05  time=223.2s
[E249] train=0.014294  val=0.016803  lr=9.98e-05  time=226.9s
[E250] train=0.013186  val=0.020489  lr=9.98e-05  time=226.2s
[E251] train=0.012447  val=0.017652  lr=9.98e-05  time=300.5s
[E252] train=0.011769  val=0.018377  lr=9.98e-05  time=224.0s
[E253] train=0.012789  val=0.015713  lr=9.98e-05  time=245.1s
[E254] train=0.011738  val=0.015817  lr=9.98e-05  time=221.7s
[E255] train=0.012452  val=0.016444  lr=9.98e-05  time=251.2s
[E256] train=0.010853  val=0.017729  lr=9.98e-05  time=220.6s
[E257] train=0.013524  val=0.014192  lr=9.98e-05  time=242.4s
[E258] train=0.013446  val=0.018259  lr=9.98e-05  time=274.9s
[E259] train=0.012171  val=0.017437  lr=9.98e-05  time=220.4s
[E260] train=0.014106  val=0.017006  lr=9.98e-05  time=236.1s
[E261] train=0.011963  val=0.017489  lr=9.97e-05  time=242.4s
[E262] train=0.013311  val=0.018495  lr=9.97e-05  time=225.7s
[E263] train=0.012535  val=0.018471  lr=9.97e-05  time=260.8s
[E264] train=0.014397  val=0.018093  lr=9.97e-05  time=255.2s
[E265] train=0.012801  val=0.017131  lr=9.97e-05  time=223.0s
[E266] train=0.013442  val=0.017523  lr=9.97e-05  time=220.6s
[E267] train=0.013345  val=0.020868  lr=9.97e-05  time=255.8s
[E268] train=0.014611  val=0.016565  lr=9.97e-05  time=220.2s
[E269] train=0.011936  val=0.016095  lr=9.97e-05  time=284.0s
[E270] train=0.013936  val=0.020178  lr=9.97e-05  time=295.4s
[E271] train=0.012896  val=0.018889  lr=9.97e-05  time=294.9s
[E272] train=0.013066  val=0.017774  lr=9.97e-05  time=294.3s
[E273] train=0.011224  val=0.018615  lr=9.97e-05  time=293.9s
[E274] train=0.011003  val=0.016199  lr=9.97e-05  time=296.4s
[E275] train=0.014135  val=0.019041  lr=9.97e-05  time=295.6s
[E276] train=0.014367  val=0.018877  lr=9.97e-05  time=296.3s
[E277] train=0.013513  val=0.017094  lr=9.97e-05  time=296.2s
[E278] train=0.012806  val=0.017289  lr=9.97e-05  time=296.2s
[E279] train=0.012728  val=0.016079  lr=9.97e-05  time=296.0s
[E280] train=0.012695  val=0.018994  lr=9.97e-05  time=297.6s
[E281] train=0.010916  val=0.015262  lr=9.97e-05  time=296.4s
[E282] train=0.012521  val=0.022276  lr=9.97e-05  time=295.7s
[E283] train=0.013803  val=0.015683  lr=9.97e-05  time=278.0s
[E284] train=0.011096  val=0.021507  lr=9.97e-05  time=228.7s
[E285] train=0.012206  val=0.020394  lr=9.97e-05  time=303.3s
[E286] train=0.010878  val=0.021949  lr=9.97e-05  time=296.5s
[E287] train=0.011416  val=0.017037  lr=9.97e-05  time=297.1s
[E288] train=0.010380  val=0.017841  lr=9.97e-05  time=296.8s
[E289] train=0.010358  val=0.015811  lr=9.97e-05  time=298.8s
[E290] train=0.012217  val=0.020019  lr=9.97e-05  time=298.3s
[E291] train=0.013498  val=0.020620  lr=9.97e-05  time=300.1s
[E292] train=0.011637  val=0.020509  lr=9.97e-05  time=301.3s
[E293] train=0.011401  val=0.015361  lr=9.97e-05  time=298.9s
[E294] train=0.012327  val=0.016248  lr=9.97e-05  time=298.1s
[E295] train=0.011136  val=0.017355  lr=9.97e-05  time=299.7s
[E296] train=0.012488  val=0.020390  lr=9.97e-05  time=300.1s
[E297] train=0.013839  val=0.018430  lr=9.97e-05  time=298.6s
[E298] train=0.011705  val=0.021875  lr=9.97e-05  time=299.2s
[E299] train=0.011803  val=0.015378  lr=9.97e-05  time=300.4s
[E300] train=0.011012  val=0.015893  lr=9.97e-05  time=299.8s
[E301] train=0.011434  val=0.018587  lr=9.97e-05  time=300.5s
[E302] train=0.010268  val=0.016570  lr=9.97e-05  time=299.9s
[E303] train=0.011034  val=0.017509  lr=9.97e-05  time=299.1s
[E304] train=0.010692  val=0.014467  lr=9.97e-05  time=299.2s
[E305] train=0.013951  val=0.020528  lr=9.97e-05  time=299.3s
[E306] train=0.011336  val=0.015750  lr=9.97e-05  time=299.3s
[E307] train=0.013397  val=0.019394  lr=9.97e-05  time=298.8s
[E308] train=0.014155  val=0.017283  lr=9.96e-05  time=298.9s
[E309] train=0.013028  val=0.018623  lr=9.96e-05  time=297.9s
[E310] train=0.011269  val=0.016712  lr=9.96e-05  time=299.8s
[E311] train=0.012718  val=0.019253  lr=9.96e-05  time=299.5s
[E312] train=0.011608  val=0.020176  lr=9.96e-05  time=298.2s
[E313] train=0.010682  val=0.018662  lr=9.96e-05  time=296.7s
[E314] train=0.012385  val=0.017951  lr=9.96e-05  time=297.9s
[E315] train=0.011087  val=0.018685  lr=9.96e-05  time=299.0s
[E316] train=0.010778  val=0.021841  lr=9.96e-05  time=297.5s
[E317] train=0.011109  val=0.013480  lr=9.96e-05  time=296.2s
[E318] train=0.013686  val=0.018849  lr=9.96e-05  time=298.3s
[E319] train=0.010456  val=0.019158  lr=9.96e-05  time=296.8s
[E320] train=0.010371  val=0.020356  lr=9.96e-05  time=296.6s
[E321] train=0.012164  val=0.019941  lr=9.96e-05  time=297.6s
[E322] train=0.012807  val=0.022980  lr=9.96e-05  time=297.5s
[E323] train=0.010466  val=0.017242  lr=9.96e-05  time=298.4s
[E324] train=0.010774  val=0.021196  lr=9.96e-05  time=297.4s
[E325] train=0.010077  val=0.017538  lr=9.96e-05  time=297.1s
[E326] train=0.010464  val=0.016321  lr=9.96e-05  time=297.7s
[E327] train=0.012357  val=0.022224  lr=9.96e-05  time=298.8s
[E328] train=0.012151  val=0.016095  lr=9.96e-05  time=298.6s
[E329] train=0.013597  val=0.017518  lr=9.96e-05  time=299.0s
[E330] train=0.010741  val=0.018369  lr=9.96e-05  time=299.6s
[E331] train=0.010228  val=0.017141  lr=9.96e-05  time=298.7s
[E332] train=0.009467  val=0.016705  lr=9.96e-05  time=299.5s
[E333] train=0.013640  val=0.017194  lr=9.96e-05  time=299.6s
[E334] train=0.011945  val=0.017302  lr=9.96e-05  time=299.0s
[E335] train=0.010600  val=0.018382  lr=9.96e-05  time=299.4s
[E336] train=0.009937  val=0.018348  lr=9.96e-05  time=299.5s
[E337] train=0.012021  val=0.019064  lr=9.96e-05  time=299.2s
[E338] train=0.012393  val=0.015445  lr=9.96e-05  time=298.9s
[E339] train=0.013199  val=0.017415  lr=9.96e-05  time=300.0s
[E340] train=0.015209  val=0.019770  lr=9.96e-05  time=299.8s
[E341] train=0.012968  val=0.015692  lr=9.96e-05  time=299.5s
[E342] train=0.011954  val=0.018564  lr=9.96e-05  time=299.5s
[E343] train=0.011410  val=0.019487  lr=9.96e-05  time=293.1s
[E344] train=0.010869  val=0.024704  lr=9.96e-05  time=298.1s
[E345] train=0.010049  val=0.020702  lr=9.96e-05  time=299.0s
[E346] train=0.010132  val=0.018520  lr=9.96e-05  time=298.7s
[E347] train=0.010308  val=0.019667  lr=9.96e-05  time=298.1s
[E348] train=0.010383  val=0.017499  lr=9.95e-05  time=298.8s
[E349] train=0.009554  val=0.020611  lr=9.95e-05  time=298.3s
[E350] train=0.010892  val=0.019564  lr=9.95e-05  time=299.9s
[E351] train=0.012267  val=0.018835  lr=9.95e-05  time=298.5s
[E352] train=0.011434  val=0.016369  lr=9.95e-05  time=298.6s
[E353] train=0.011830  val=0.018120  lr=9.95e-05  time=298.1s
[E354] train=0.009988  val=0.015604  lr=9.95e-05  time=296.4s
[E355] train=0.010468  val=0.017525  lr=9.95e-05  time=298.2s
[E356] train=0.009716  val=0.016706  lr=9.95e-05  time=296.4s
[E357] train=0.009960  val=0.016822  lr=9.95e-05  time=296.7s
[E358] train=0.009246  val=0.014550  lr=9.95e-05  time=296.3s
[E359] train=0.009956  val=0.015942  lr=9.95e-05  time=295.9s
[E360] train=0.011648  val=0.015116  lr=9.95e-05  time=296.5s
[E361] train=0.013002  val=0.016149  lr=9.95e-05  time=296.4s
[E362] train=0.012232  val=0.019637  lr=9.95e-05  time=297.4s
[E363] train=0.008933  val=0.016136  lr=9.95e-05  time=296.9s
[E364] train=0.010264  val=0.019689  lr=9.95e-05  time=297.8s
[E365] train=0.010302  val=0.017138  lr=9.95e-05  time=297.7s
[E366] train=0.011111  val=0.017147  lr=9.95e-05  time=297.9s
[E367] train=0.009747  val=0.018052  lr=9.95e-05  time=297.2s
[E368] train=0.009564  val=0.017980  lr=9.95e-05  time=297.2s
[E369] train=0.010453  val=0.018945  lr=9.95e-05  time=298.2s
[E370] train=0.009990  val=0.017505  lr=9.95e-05  time=299.3s
[E371] train=0.011597  val=0.017669  lr=9.95e-05  time=297.3s
[E372] train=0.010915  val=0.019683  lr=9.95e-05  time=298.6s
[E373] train=0.011137  val=0.017185  lr=9.95e-05  time=298.0s
[E374] train=0.011937  val=0.019879  lr=9.95e-05  time=298.0s
[E375] train=0.010053  val=0.017853  lr=9.95e-05  time=297.0s
[E376] train=0.009076  val=0.017807  lr=9.95e-05  time=298.9s
[E377] train=0.009254  val=0.018000  lr=9.95e-05  time=298.0s
[E378] train=0.009118  val=0.015910  lr=9.95e-05  time=299.7s
[E379] train=0.008795  val=0.017140  lr=9.95e-05  time=298.4s
[E380] train=0.009619  val=0.017104  lr=9.95e-05  time=299.7s
[E381] train=0.009574  val=0.014436  lr=9.95e-05  time=300.1s
[E382] train=0.009538  val=0.017758  lr=9.95e-05  time=298.9s
[E383] train=0.010192  val=0.015626  lr=9.95e-05  time=299.5s
[E384] train=0.010386  val=0.017336  lr=9.94e-05  time=299.8s
[E385] train=0.011344  val=0.016365  lr=9.94e-05  time=298.2s
[E386] train=0.009600  val=0.017793  lr=9.94e-05  time=297.8s
[E387] train=0.009334  val=0.016689  lr=9.94e-05  time=299.3s
[E388] train=0.009479  val=0.020948  lr=9.94e-05  time=298.2s
[E389] train=0.008914  val=0.015977  lr=9.94e-05  time=298.0s
[E390] train=0.009164  val=0.022136  lr=9.94e-05  time=297.9s
[E391] train=0.010140  val=0.019555  lr=9.94e-05  time=297.2s
[E392] train=0.011260  val=0.017245  lr=9.94e-05  time=298.2s
[E393] train=0.010777  val=0.017472  lr=9.94e-05  time=298.5s
[E394] train=0.009804  val=0.015138  lr=9.94e-05  time=297.3s
[E395] train=0.009384  val=0.015581  lr=9.94e-05  time=296.0s
[E396] train=0.007999  val=0.021673  lr=9.94e-05  time=295.2s
[E397] train=0.008935  val=0.019576  lr=9.94e-05  time=297.5s
[E398] train=0.009079  val=0.018118  lr=9.94e-05  time=297.8s
[E399] train=0.010768  val=0.018996  lr=9.94e-05  time=297.1s
[E400] train=0.010623  val=0.018876  lr=9.94e-05  time=298.0s
[E401] train=0.008977  val=0.019053  lr=9.94e-05  time=297.5s
[E402] train=0.008963  val=0.017251  lr=9.94e-05  time=298.1s
[E403] train=0.009133  val=0.016530  lr=9.94e-05  time=298.2s
[E404] train=0.010222  val=0.017442  lr=9.94e-05  time=298.6s
[E405] train=0.013560  val=0.014188  lr=9.94e-05  time=298.3s
[E406] train=0.012537  val=0.018202  lr=9.94e-05  time=298.2s
[E407] train=0.012310  val=0.016666  lr=9.94e-05  time=298.1s
[E408] train=0.009516  val=0.016984  lr=9.94e-05  time=298.4s
[E409] train=0.009407  val=0.015967  lr=9.94e-05  time=298.8s
[E410] train=0.010029  val=0.016645  lr=9.94e-05  time=299.3s
[E411] train=0.009223  val=0.019631  lr=9.94e-05  time=298.2s
[E412] train=0.008089  val=0.020875  lr=9.94e-05  time=299.1s
[E413] train=0.009207  val=0.014417  lr=9.94e-05  time=299.3s
[E414] train=0.008710  val=0.019775  lr=9.94e-05  time=298.3s
[E415] train=0.008757  val=0.017690  lr=9.94e-05  time=297.1s
[E416] train=0.008894  val=0.019528  lr=9.94e-05  time=299.3s
[E417] train=0.009514  val=0.017183  lr=9.93e-05  time=299.1s
[E418] train=0.008706  val=0.020180  lr=9.93e-05  time=298.0s
[E419] train=0.009654  val=0.019476  lr=9.93e-05  time=299.1s
[E420] train=0.009244  val=0.015021  lr=9.93e-05  time=297.4s
[E421] train=0.008755  val=0.017637  lr=9.93e-05  time=297.0s
[E422] train=0.009365  val=0.017032  lr=9.93e-05  time=297.1s
[E423] train=0.009780  val=0.016847  lr=9.93e-05  time=298.4s
[E424] train=0.009727  val=0.015988  lr=9.93e-05  time=296.5s
[E425] train=0.008841  val=0.017763  lr=9.93e-05  time=297.4s
[E426] train=0.011400  val=0.015979  lr=9.93e-05  time=297.2s
[E427] train=0.010286  val=0.017302  lr=9.93e-05  time=297.3s
[E428] train=0.008901  val=0.015274  lr=9.93e-05  time=297.4s
[E429] train=0.009472  val=0.013846  lr=9.93e-05  time=296.5s
[E430] train=0.009814  val=0.016701  lr=9.93e-05  time=296.3s
[E431] train=0.009069  val=0.016701  lr=9.93e-05  time=295.8s
[E432] train=0.011895  val=0.015889  lr=9.93e-05  time=295.6s
[E433] train=0.011501  val=0.017501  lr=9.93e-05  time=296.0s
[E434] train=0.009507  val=0.019052  lr=9.93e-05  time=296.1s
[E435] train=0.008484  val=0.018069  lr=9.93e-05  time=295.8s
[E436] train=0.009738  val=0.017737  lr=9.93e-05  time=296.8s
[E437] train=0.010345  val=0.017622  lr=9.93e-05  time=296.5s
[E438] train=0.010907  val=0.018387  lr=9.93e-05  time=305.3s
[E439] train=0.010375  val=0.017307  lr=9.93e-05  time=301.4s
[E440] train=0.010497  val=0.017360  lr=9.93e-05  time=296.2s
[E441] train=0.010232  val=0.015580  lr=9.93e-05  time=298.3s
[E442] train=0.010679  val=0.018835  lr=9.93e-05  time=298.6s
[E443] train=0.008928  val=0.017002  lr=9.93e-05  time=298.1s
[E444] train=0.010465  val=0.016737  lr=9.93e-05  time=299.4s
[E445] train=0.009484  val=0.016883  lr=9.93e-05  time=297.2s
[E446] train=0.008489  val=0.019602  lr=9.93e-05  time=297.8s
[E447] train=0.008590  val=0.017685  lr=9.93e-05  time=298.7s
[E448] train=0.008964  val=0.016185  lr=9.92e-05  time=299.0s
[E449] train=0.010981  val=0.016786  lr=9.92e-05  time=298.1s
[E450] train=0.008677  val=0.015461  lr=9.92e-05  time=299.2s
[E451] train=0.008786  val=0.016617  lr=9.92e-05  time=297.8s
[E452] train=0.007558  val=0.018475  lr=9.92e-05  time=299.3s
[E453] train=0.008293  val=0.017297  lr=9.92e-05  time=297.4s
[E454] train=0.010315  val=0.018383  lr=9.92e-05  time=298.5s
[E455] train=0.011058  val=0.014504  lr=9.92e-05  time=296.8s
[E456] train=0.008662  val=0.014957  lr=9.92e-05  time=297.0s
[E457] train=0.008386  val=0.018577  lr=9.92e-05  time=296.2s
[E458] train=0.008325  val=0.015445  lr=9.92e-05  time=296.0s
[E459] train=0.009409  val=0.018558  lr=9.92e-05  time=297.2s
[E460] train=0.008281  val=0.018159  lr=9.92e-05  time=296.4s
[E461] train=0.009886  val=0.018390  lr=9.92e-05  time=296.8s
[E462] train=0.009493  val=0.018117  lr=9.92e-05  time=296.5s
[E463] train=0.010370  val=0.018141  lr=9.92e-05  time=296.6s
[E464] train=0.007856  val=0.018015  lr=9.92e-05  time=295.5s
[E465] train=0.008003  val=0.014737  lr=9.92e-05  time=295.5s
[E466] train=0.009286  val=0.018798  lr=9.92e-05  time=296.3s
[E467] train=0.010894  val=0.015006  lr=9.92e-05  time=295.4s
[E468] train=0.008941  val=0.018049  lr=9.92e-05  time=295.1s
[E469] train=0.009835  val=0.017772  lr=9.92e-05  time=297.2s
[E470] train=0.008373  val=0.016739  lr=9.92e-05  time=295.8s
[E471] train=0.009808  val=0.014656  lr=9.92e-05  time=296.5s
[E472] train=0.011240  val=0.017397  lr=9.92e-05  time=297.5s
[E473] train=0.009286  val=0.015204  lr=9.92e-05  time=296.8s
[E474] train=0.008484  val=0.016160  lr=9.92e-05  time=297.6s
[E475] train=0.008416  val=0.017810  lr=9.92e-05  time=298.5s
[E476] train=0.008607  val=0.014484  lr=9.91e-05  time=298.6s
[E477] train=0.007470  val=0.019181  lr=9.91e-05  time=298.4s
[E478] train=0.009695  val=0.016584  lr=9.91e-05  time=298.0s
[E479] train=0.010805  val=0.015950  lr=9.91e-05  time=297.8s
[E480] train=0.008734  val=0.019361  lr=9.91e-05  time=299.0s
[E481] train=0.007615  val=0.017360  lr=9.91e-05  time=298.6s
[E482] train=0.007872  val=0.015105  lr=9.91e-05  time=298.2s
[E483] train=0.008850  val=0.015340  lr=9.91e-05  time=298.2s
[E484] train=0.009889  val=0.016662  lr=9.91e-05  time=298.4s
[E485] train=0.009501  val=0.018386  lr=9.91e-05  time=297.5s
[E486] train=0.008345  val=0.015662  lr=9.91e-05  time=298.0s
[E487] train=0.007666  val=0.016824  lr=9.91e-05  time=298.3s
[E488] train=0.008170  val=0.017515  lr=9.91e-05  time=298.8s
[E489] train=0.008457  val=0.017471  lr=9.91e-05  time=298.0s
[E490] train=0.007831  val=0.015444  lr=9.91e-05  time=299.2s
[E491] train=0.008847  val=0.018618  lr=9.91e-05  time=299.4s
[E492] train=0.007376  val=0.018047  lr=9.91e-05  time=299.4s
[E493] train=0.008265  val=0.020818  lr=9.91e-05  time=297.7s
[E494] train=0.008335  val=0.014460  lr=9.91e-05  time=298.8s
[E495] train=0.008899  val=0.019360  lr=9.91e-05  time=298.9s
[E496] train=0.008177  val=0.016948  lr=9.91e-05  time=299.0s
[E497] train=0.008842  val=0.019189  lr=9.91e-05  time=299.2s
[E498] train=0.010218  val=0.016639  lr=9.91e-05  time=299.7s
[E499] train=0.008614  val=0.019292  lr=9.91e-05  time=298.2s
[E500] train=0.008492  val=0.016172  lr=9.91e-05  time=298.8s
[E501] train=0.009373  val=0.020063  lr=9.91e-05  time=297.3s
[E502] train=0.007638  val=0.015618  lr=9.91e-05  time=298.6s
[E503] train=0.007837  val=0.013496  lr=9.90e-05  time=297.8s
[E504] train=0.009639  val=0.019206  lr=9.90e-05  time=305.0s
[E505] train=0.009459  val=0.019363  lr=9.90e-05  time=305.1s
[E506] train=0.010757  val=0.016842  lr=9.90e-05  time=305.6s
[E507] train=0.008407  val=0.019115  lr=9.90e-05  time=299.8s
[E508] train=0.007870  val=0.015121  lr=9.90e-05  time=308.0s
[E509] train=0.007970  val=0.015224  lr=9.90e-05  time=298.2s
[E510] train=0.009035  val=0.017443  lr=9.90e-05  time=296.2s
[E511] train=0.008004  val=0.018733  lr=9.90e-05  time=298.2s
[E512] train=0.008906  val=0.015734  lr=9.90e-05  time=299.5s
[E513] train=0.007519  val=0.015764  lr=9.90e-05  time=298.2s
[E514] train=0.007610  val=0.015732  lr=9.90e-05  time=297.9s
[E515] train=0.008093  val=0.018309  lr=9.90e-05  time=297.4s
[E516] train=0.008139  val=0.015325  lr=9.90e-05  time=296.4s
[E517] train=0.010037  val=0.018452  lr=9.90e-05  time=296.1s
[E518] train=0.008135  val=0.019429  lr=9.90e-05  time=296.3s
[E519] train=0.007548  val=0.014771  lr=9.90e-05  time=296.1s
[E520] train=0.008483  val=0.019239  lr=9.90e-05  time=296.2s
[E521] train=0.008431  val=0.015991  lr=9.90e-05  time=295.8s
[E522] train=0.009341  val=0.018352  lr=9.90e-05  time=297.5s
[E523] train=0.007405  val=0.013962  lr=9.90e-05  time=295.6s
[E524] train=0.007099  val=0.019313  lr=9.90e-05  time=296.8s
[E525] train=0.008769  val=0.018776  lr=9.90e-05  time=296.3s
[E526] train=0.009283  val=0.015794  lr=9.90e-05  time=295.8s
[E527] train=0.007408  val=0.017482  lr=9.90e-05  time=297.5s
[E528] train=0.007835  val=0.014896  lr=9.90e-05  time=296.4s
[E529] train=0.007645  val=0.016553  lr=9.89e-05  time=296.0s
[E530] train=0.007477  val=0.014967  lr=9.89e-05  time=296.3s
[E531] train=0.008736  val=0.016234  lr=9.89e-05  time=294.9s
[E532] train=0.010956  val=0.017310  lr=9.89e-05  time=295.6s
[E533] train=0.008784  val=0.016477  lr=9.89e-05  time=295.4s
[E534] train=0.007755  val=0.016737  lr=9.89e-05  time=305.9s
[E535] train=0.008727  val=0.016067  lr=9.89e-05  time=299.5s
[E536] train=0.008842  val=0.015655  lr=9.89e-05  time=297.1s
[E537] train=0.007010  val=0.014859  lr=9.89e-05  time=295.9s
[E538] train=0.007223  val=0.015912  lr=9.89e-05  time=231.3s
[E539] train=0.007765  val=0.016498  lr=9.89e-05  time=264.9s
[E540] train=0.007533  val=0.018556  lr=9.89e-05  time=295.2s
[E541] train=0.006769  val=0.014948  lr=9.89e-05  time=297.0s
[E542] train=0.007018  val=0.016702  lr=9.89e-05  time=297.0s
[E543] train=0.007518  val=0.014584  lr=9.89e-05  time=296.2s
[E544] train=0.007246  val=0.017355  lr=9.89e-05  time=295.4s
[E545] train=0.008374  val=0.017390  lr=9.89e-05  time=294.8s
[E546] train=0.008921  val=0.016896  lr=9.89e-05  time=296.8s
[E547] train=0.008443  val=0.016317  lr=9.89e-05  time=300.1s
[E548] train=0.007693  val=0.017207  lr=9.89e-05  time=296.7s
[E549] train=0.006629  val=0.018740  lr=9.89e-05  time=295.6s
[E550] train=0.008325  val=0.014387  lr=9.89e-05  time=295.5s
[E551] train=0.007362  val=0.016554  lr=9.89e-05  time=297.1s
[E552] train=0.007620  val=0.017552  lr=9.89e-05  time=296.6s
[E553] train=0.009152  val=0.016815  lr=9.88e-05  time=296.2s
[E554] train=0.007832  val=0.018817  lr=9.88e-05  time=296.8s
"""
# NOTE: Replace TRAINING_LOG content above, OR pass --log to read from a file.


def parse_log(text: str):
    """Extract (epoch, train_loss, val_loss, lr) tuples from a training log."""
    # Matches: [E000] train=1.016982  val=0.988713  lr=1.42e-05  time=265.2s
    pattern = re.compile(
        r"\[E(\d+)\]\s+train=([\d.]+)\s+val=([\d.]+)\s+lr=([\d.e+-]+)"
    )
    epochs, train_losses, val_losses, lrs = [], [], [], []
    for m in pattern.finditer(text):
        epochs.append(int(m.group(1)))
        train_losses.append(float(m.group(2)))
        val_losses.append(float(m.group(3)))
        lrs.append(float(m.group(4)))
    return (np.array(epochs), np.array(train_losses),
            np.array(val_losses), np.array(lrs))


def plot_curves(epochs, train, val, lrs, out_path="training_loss.png"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(epochs, train, label="train_loss", color="tab:blue", alpha=0.8, linewidth=1.2)
    ax.plot(epochs, val, label="val_loss", color="tab:orange", alpha=0.8, linewidth=1.2)

    # Mark best val
    best_idx = int(np.argmin(val))
    ax.scatter(epochs[best_idx], val[best_idx], color="red", zorder=5,
               s=60, label=f"best val={val[best_idx]:.6f} @E{epochs[best_idx]}")
    ax.axvline(epochs[best_idx], color="red", linestyle="--", alpha=0.3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Saved plot to {out_path}")

    # Print summary
    print(f"\n=== Summary over {len(epochs)} epochs ===")
    print(f"  Final train_loss: {train[-1]:.6f}")
    print(f"  Final val_loss:   {val[-1]:.6f}")
    print(f"  Best val_loss:    {val[best_idx]:.6f} at epoch {epochs[best_idx]}")
    print(f"  Min train_loss:   {train.min():.6f} at epoch {epochs[int(np.argmin(train))]}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default=None,
                        help="Path to training log file (txt). If not given, use TRAINING_LOG in this script.")
    parser.add_argument("--out", type=str, default="training_loss.png",
                        help="Output plot path")
    args = parser.parse_args()

    if args.log:
        text = Path(args.log).read_text(encoding="utf-8", errors="ignore")
    else:
        text = TRAINING_LOG

    epochs, train, val, lrs = parse_log(text)
    if len(epochs) == 0:
        print("[ERROR] No training entries parsed from log.")
        sys.exit(1)

    print(f"[INFO] Parsed {len(epochs)} epoch entries")
    plot_curves(epochs, train, val, lrs, args.out)


if __name__ == "__main__":
    main()
