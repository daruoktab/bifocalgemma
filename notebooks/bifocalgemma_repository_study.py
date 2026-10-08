# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo==0.25.1", "torch==2.11.0", "transformers==5.18.0", "safetensors==0.8.0", "pyarrow==25.0.0", "pytest==8.4.2"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        "# BifocalGemma · Repository study\nPrepare → SFT → generate → native score. Default actions disabled; fixture is synthetic. Eager generation has no temporal cache. Native Python execution is opt-in and is not a security sandbox."
    )
    return (mo,)


@app.cell
def _():
    # CONTROL CENTER: choose one action/run at a time on a shared GPU.
    SOURCE = "fixture"
    EXPERIMENT = "BifocalGemma-E2B-E2B"
    ROWS_FILE = ""
    PREPARED_DIR = "outputs/repository_study/prepared"
    RUN_DIR = "outputs/repository_study/training"
    PREDICTIONS_FILE = "outputs/repository_study/predictions.jsonl"
    SCORE_FILE = "outputs/repository_study/score.json"
    TOKENIZER_CACHE = "outputs/repository_tokenizers"
    MODEL_CACHE = "outputs/hf_cache"
    ADAPTER_CHECKPOINT = ""
    PYTHON_EVAL = ""
    RUN_PREPARE = False
    RUN_TRAINING = False
    RUN_GENERATE = False
    RUN_SCORE = False
    VALIDATE_GOLD = False
    ALLOW_CODE_EXECUTION = False
    FETCH_SOURCE_ROWS = False
    FETCH_TOKENIZERS = False
    ALLOW_GIT_NETWORK = False
    CONTEXT_LIMIT = 4096
    TRAIN_DECODER_LIMIT = 2048
    MAX_NEW_TOKENS = 512
    GENERATION_DECODER_LIMIT = 8192
    RAW_LIMIT = 1000
    TASK_LIMIT = 32
    PER_REPO_LIMIT = 8
    MAX_STEPS = 100
    EPOCHS = 1
    ACCUMULATION = 4
    HEADROOM_GIB = 16
    return (
        SOURCE,
        EXPERIMENT,
        ROWS_FILE,
        PREPARED_DIR,
        RUN_DIR,
        PREDICTIONS_FILE,
        SCORE_FILE,
        TOKENIZER_CACHE,
        MODEL_CACHE,
        ADAPTER_CHECKPOINT,
        PYTHON_EVAL,
        RUN_PREPARE,
        RUN_TRAINING,
        RUN_GENERATE,
        RUN_SCORE,
        VALIDATE_GOLD,
        ALLOW_CODE_EXECUTION,
        FETCH_SOURCE_ROWS,
        FETCH_TOKENIZERS,
        ALLOW_GIT_NETWORK,
        CONTEXT_LIMIT,
        TRAIN_DECODER_LIMIT,
        MAX_NEW_TOKENS,
        GENERATION_DECODER_LIMIT,
        RAW_LIMIT,
        TASK_LIMIT,
        PER_REPO_LIMIT,
        MAX_STEPS,
        EPOCHS,
        ACCUMULATION,
        HEADROOM_GIB,
    )


@app.cell
def _(mo):
    # BUNDLED SOURCE: generated from src/configs/fixtures; no weights or real dataset.
    import base64
    import hashlib
    import io
    import json
    import os
    import sys
    import zipfile
    from pathlib import Path
    from types import SimpleNamespace

    import torch
    from huggingface_hub import HfApi, hf_hub_download

    bundle_payload = (
        "P)h>@6aWAK2mk;8Apq&FOCQz&004*$000;O003ieZf0p`b1!9haAk67ZDnqBb1rIgZ*JvP!HU~35WO$#KLnq<)RITG)nlM6z4um%"
        "8A~ItK(>RdY)hB^du2Omv)iOg2n$Jy5i&EQr+Io0>FM@nu}B8>KrL2}<Okb#$%h4dH^qZT+u5$EJSn@Pm`JpZ|DhU8Nah-ChhUw<"
        "Ft0S|2AVVUeMR23)P*4?zKpTO;%S~@y2ExtTw2pBbobP@c=wSn$1!71Wz?x&laH0Zm&RxtW3IA#1ZeucarC~K0O0xZSnBD~ZfGT%"
        "OJnjHgh@+=Ac=U9WmpQS#*b?$G{f9H0~qX&snNfHdpE^59ac3Clx*`q|1i*WJcBzM4wA@-bI6&=1(lMCi~*PubzP*C)urM}$r1z>"
        "N0HbxS;c+#P{0tPjB)>JY<sM4$v32TbPATL?MZCGzV%*@+6TX;oyyl;HJ{rdXU&8j&{MUVx}Q>WX4~{@K7-{rPVyNX_8BmS$cnlE"
        "i6{lvhGjx(t`%XCNmif$&=U7c^b+{Zd<OsY8Bt0mPkEMeri>9O=t7h{=Nc%5QVJAk0QpfQ{x_fT(|vaN_t~}i?1e?n!f0^%!+H4^"
        "+O<8L=V7$_o#XxVd9TH~wR2Z4oWk{h?lOggHxupxuJ-d4Zk)c}-o*JAP)h>@6aWAK2mk;8ApknoanoP~004~(0012T003ieZf0p`"
        "b1!mbaBp*IbZ>HbUvqSIWO*)Xb8l{qR!fiEHVnQ8$bT^Q9Czf0{Af=_n-pn#YKry{6owwuj<UApNb=51g8cVVw%3^?UT;p4DDo32"
        "lArYP`Ligxxd-dC>2H3Fk{6R;80}yjpy0*=&a^l75)0FNIQz)?QC@zEB=am{vePaYN7_Zb@ezyE*tSv7Wu5Ms4|@Wv*Gh|Nbu*vg"
        "%xDETJmX_nqeLUT?c46~=HHj`D}4qV|81G><0K(}1Yc^%RFV}1hg8%oVQEHWnrA#Gq{?WX<5oo}N#+v}F#!E`nE`PO7@XHRNJOeR"
        "2VNJfRDvr`vy4|nr4-ka)`G7XJ#am`3Bm%0UK|5VRST&&<2mCAmzAQdpcUt7lFEb^S&`-{S5=klz{LNg0VyGQl}R9q5(QK;n$tQ@"
        "OM;#hqDU~DGGVGzE7D)-9Mm$#Z$|FG2dEUPgj7nQy+Q{Br$tq8CRMPnSyhpW7KNlM)GKF{kAHss%i%9f^UE`c0F$R&VoZdTDG4)~"
        "C_;E$38*RsKq)O{0V(Xj#IJkj?I<R<C=+Q838kC_!;(PBD#9|#^NKOSIcGv)yer<HzmqDVhhO3T<vnYsODdEqb0Vu!QN`+#7KkL3"
        "k~zohrFETVNtz_$i_A~AJl(kKv_HnfB|w#>V7w-j$~+DEqavlE!rM`GR+74qJjp1nK1VlL-3#!hkNb8wtiS;a@(S>>5~vjjNjaBA"
        "j*C3WnV_{~wMwC=RdExbKFxI>8meKfX6}pP_rSXMqqWfc!_oWUJmL4kNLbb{HZ-Mn!dUsk9GYdr;6Gdow`>M$PA%AAuuNmono+JB"
        "TSiRF*<_l;OfQlxT?_9=aNbQPIQ4nGr7Seo$)rlo60oBz(#nH$%fJlKGmXjg$BD9TX2DnTElY<i#%^J$12+zgV-l{#TUPJTc{^Fm"
        "vj@SJZfhiSQ)SGv2f=0<)&TqwSbJ+M)@bJrI{|R*h0*=M2IN!AmW+G=dpioDx3;%XPQbIJU{Ld9?0FS!`#FqrOz+-#5J!?^<Ux2<"
        "*1R?14rDXE7pVPqf!fLHBc=41Q$i0%xE$tTm2S2O;FVe!yKgYQ4Po8cOfCjBI()Pa^W;jtmEMk7ckEc(no~S^btU+$)QS*u7}~a5"
        "#k*C!->0F+c+14y7m{zfZn)gWA+e#tj1|Y`7EA};YH_6!j*SO{;>%$k*3HM+#R3yv-^1eeVOn49e{ZWjPAT76+B>xI1SvjFu}`=A"
        "(=4?|!#h2<LmRui`vm#zk;1yS$4PsP6IA}2DPim_IMd!ED@-pYcEers)${ju(Ov$Q&0tzBCT<!=j?)Hzmm@U1OKEe`y)-A++%LbX"
        ")u`3@zN1Z!&-c9J!@p7Yw~d6*h;~&;@wYXr0DdjT=|kFd?A-KlYP`9FK6q6nb-HNbXJ(v~Bq%nv0YkCY!Mr`SHrje*th;TM4$p5f"
        "Y=_0&&!L$fqDdU#y+wBH@zy>)fA&96O9KQH000080000X09`3Nre*;E0PO+*02%-Q0Ap`%W@%@0FLZKYX>KlMcwudDY-KKLb8l{S"
        "Qn79VF$~?2_y=@z1v^`Zwqik5#U7c&ISjcZO`Hl^_228<ag>wVF@Dd_&oBO(&lZa&GjPexdeKA&8V|u4Oe*1fp@dn*6f8ONZS&C6"
        "IRSZj3PV%G+Gm8QS!K&32SFcxMU0RvxQqj8C^I<+Ou<hPhmZzs;HvD4a|6<mPC@N~^+9F&oPuovL+Dw7)}<?hvdWAxPetJTM>W<("
        "^5EsJidHt_YNWkP5CaUV15Xw+_v=exH5Qh~9K5^c5DLe_O7f|^;gzqL3F1{uk1^CjUhUo<H`~qjW!afA%hD&(25`NIwelIL$cwq6"
        "`%O8hSd+PSBqVP`(pcrynCUcWJfkjsvqdy}(5ESW`~HvaA3*G$FipXjsDpG-HIa;cIKIT>R!@fbRfat8#$;k04(^`1r&b4pD;>Iy"
        "I11hJI$ro7`P^>rD3=yw#CV2N1JKr$Q*bWSq_5a~_C25d08mQ<1QY-O00;m803iV9ixwt(0RRAr0ssIh0001GcwudDY-MvVa%FID"
        "b7^#Ma(Q2KbaiBTFJ*RNY+rJ3cXKXkb8l{JZBpBA;~)@yU#b5va$j0GxQT7KNu)~YOTQwLg<+`CVxU~AM63Px0!^!IwLBo0bC`2x"
        "hCh|mK2XhYl)TzQCFChYhXmiLBI{}el|yT;kcSx+CnkoBP1NJa%(BRkDVt$xMc;0BoO0~?^^WpkViOg8FZkmyh{x@A$J!x3wa!cm"
        "mocdaTu_$;FY*M8Di#$D#x-9@+=C3@XA>0%hslP9FeCVAt&+$gnT-fmMGr>f3Zj?#wNR2<i>@h(G_X0Va47^3@|tqehfjiU>=~x3"
        "%TW_T{(+CWuInG1!vV-<)j^%N!{MlbqX5eJm~7k?SNHoA&FdEYStm<DG6>ZFhIP`&z@?zVP9i@;P{Zb^a{)>+)E;vQ&z7<oN!yeW"
        "eJ36-LrRq3VroPc(1+18Cv9o4aAvvcFv!Ik7rrA&pL&nik-P35o}Qrp4?@}kk&C@7H6b5Sze{KI*G%m>%tm+3ru+O&Z&A83a6g7q"
        "VN`!_etiH?O9KQH000080000X0OeyG4Z;8b02=`S03!eZ0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-AML|SOMJ{b*MUO#l#4rp4"
        "?*Z}$0zD@v`h&h<`vEm(*1|e=YRilD@2eb&-UN~wQE#}7Y>}oLD$ml?N*znH9tL)OS+8vKG*4fVjiBD6!K!H1d<S20oQVyR14<?$"
        "Q&>tj@F}T_v;H!0L4z3K*HHF7spn5Uc?~O^KOmLJ#qiC%#Y5@?ChgI1VV)XXy;gCQz;x9Wt@!B<wz1+?dUbzu(|b5cy;rwbDg$;$"
        "Sajw8N%S08q1Nzu$`-kNtnWy=F*mOB^b1f+0|XQR000O8001EXdo9;C-~a#siva)tX#fBKWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lN"
        "c`tKrVQ_P3Z*+4nF=A$9VP-ZqI5ajnGB9H`W@b2LHZx^2IWjXfH85f^I50IeG&wkBGB{;6Vl!nnH#Rh3GGR0{H)An3Ib$_6E^2dc"
        "Zgr2{O2aS|$L|MwheNzkm|APe+P#XEktQcJGfhHrDvsg1+q4~`u&YDz|2V&M*=_(>#NgQhm^NQDMtly@pHR|c!FIFWiG)Y#hrv*`"
        "z3{HoRarMmD15DoMkwCNPUx!AYR|de*L5RHzNWRYn5NuYegRyTF$qCXpZj{W0b_a!=Or<V(mO&o?^CEjD0TD>?>~hd@!`<AY`TMV"
        "cGM#o2_(`76w80a=?n+6w{>jk{NCN#FNKw(73AYh1RebVD<_yC_()fz*0nAZUR(Y#t}K7~f1&bt-)^S)15ir?1QY-O00;m803iUp"
        "mUo)o0001s0RRAL0001GcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8;WnyG8WI1DGF)}k_F*7qYWiw(qHZ?UfWo9%r"
        "Ffn2@WH2){W@2SxIAJ(BGi7EmHeoPkIbmjHH#IOfF=I6@YIARHb&qXJ!!QuX-w*a3j`&7lDmL1#`zlt(HNDVinuO#kj^TH=X*)z="
        "ua4yZaKAfmHvl41@ZtbMn=c+CJ%;GtN%6fDyIF5B(NXzfFr00bu1b_db%9vQR@b^FSrmKf)HAmGr<bDAM6z1b+E_|c?lr#v&dZpD"
        "fZXT4o@}6)9>Qr!%%bv+$<6yZ(ture^p0*n(T;R?Xk0ekKsq_@iH!or^a17aA89(n!R&3ESUSIV*Y+#4YP85cUPaK+53q8A8G?^|"
        "L26v%GU27=ALGjMm;V<kkGJh+nm<rW0|XQR000O8001EX=dStY-~a#siva)tX#fBKWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKr"
        "VQ_P3Z*+4nGh{R|VlrenF)(FiW;HWoV>n@9H8N#6WHV$mH8(V7HDWU`G-YHlVKgvhGdMFcH!)^qV`VvJF*r9kH8MFiE^2dcZgr1s"
        "OT#b_$KMb39gg@$VQRN-cJ8ZK8Q0`OGt(p_S8)u#yG`363VU@V|A+hCWw`(lk%AWo5ZZikAL%(ne<I0`Yq1*j1`{2n?>fWTwyxBs"
        "&{|<pm5tnQOWIY`G=#fiTfE54{#BAHOPtf%SV}|gEx!OR(-?$++~&TPY@nE)!g)%JqV$f*jr$a7K&6h}(fuddo*oXh%ceU>XU8qE"
        "k-(Tfpq&0A4QDtQy|oil=lAN?eu-B07TL#}2s-)>W==3&@R6@bwX0nwyte#foLT<z|3c;QzFZ9R2T)4`1QY-O00;m803iS_E7hIh"
        "0001s0RRAL0001GcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8>G+|_9GchtXWnwfjGB!A7F=I3`Vm2^2WHvA{WH4nk"
        "H92NEI59FdVm4%9Vlp@|G%z?gF=An2VKF&oGBh$SYIARHb&uU@!ypvL?+bm0gS*+F?Us+lq_47!qUK2KjEb7EQugkPNy^xuR|oxn"
        "oZq=_Hvq(=v*ZXws;@YDe0JWRQN(>tcC*e|z>~1UpklJ+Lh=`yl{Di;MXBTkd#$Uc5KT>UUW<}tm25a$(^6>+Q|>Lj0ItiJ1jk~Z"
        "`g+t3eK@$wl9)wdO++*A>revgx|22d_{sE$Psi3I(*uNyi9JRoAhPfQ<?`=wI>W&1Z4FsEzjycc3#P@WS+xExg7kKPl@rw9tdBRO"
        "*0d%O-dg@Kt}K7~f1%QN-)^S)15ir?1QY-O00;m803iSth}c2j0001s0RRAL0001GcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2T"
        "aC2yHbaO8?I5se2Vqq{fF*#;6V>K}}F*7x0W@a^HFk~_`W;J3sV=*^0GdMS4I5{>sWH4o6V>CHtVmW0qWH~WpI5asfYIARHb&pGG"
        "!!Qtr_krADP&X~4X-bM^<tlLrdW^7ASrYQNG=$#0mhDhl$ZC-O$9!{LuK-xa6xjiobzd}Od`UT;QPcCDZ6;mG9#1-Uougt?l~7#0"
        ")@HAEvKC5Ls06oM3Tq8FZ#Bn?i=A20)_Ls5+_Ag>uJag$B&jWZE&7DH-=@o)m_!`|VVL$Av>=T+#ek2W(hvA_XhJbPKz|9;A~_8t"
        "`w!srf5!2Q0F$=~Z0`Kt+}p3E*MpZ7^IZg+V+RW-xGu#^H>4&sp%C6${xL2rfBAo*%6MO|#`yzKO9KQH000080000X0L%vnVc-A&"
        "0Ez(s0BHaK0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFEuz~WMwpAVP<79GC43|GGt<8VlZQ5G&nIZHZo>8Gi5S3"
        "Vq-TpIWsk8V`eckGB##6G%+zZVlgu|V`XDFIWB5*Z*FyuQA@)x6ouah`wy4+qA+#VlC}FQR>o~|LxX7&l2jbS|8CNDh{B!@$-U>C"
        "?_RbW02atQb_AyMSDFGn4`k${n-bg2;~pcOgd0bl%GL|-D^*ubD^b#Gy>C(SUi7HyTFC?F>d-W;sQBtu+EC2CxAFqGERqKv(@^qq"
        "o2L+;{CRQAL*Z;9JNHRafKr~ErTb4b6FnR|TbAx1o^2W^X#ol22ke*sK=}#_^KNI^lKH*6)i2P9$zXEfCV~oXgq0KY=v_!xq|SD>"
        "5MERMF|I6s`G28`yl*#Ie*jQR0|XQR000O8001EXGdt}O-~a#siva)tX#fBKWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3"
        "Z*+4nHa0OZHDx$uWiw+rGBP<fVly!^H(_NsFfukVGdE>1W;rusGB;%~FgRp0GGsM5I5=ftW@I;FWi(?jH!x&3E^2dcZgr2{YQrED"
        "$L|Y$hl9J>pzCUCE$gc+qgXu>H=|-s%1YV0FD5BtgI*o<|8aijx?BLTj484MFzddk&-jvZ{6J058@3wtlj`uOW7|3^HnkH%Szc|W"
        "thD3`H&Tj|FoKKSUT$AaCD80GpVQWP?1tQ1c>!FfF$hUfQ~Db833InjmnkudItIcp?lWjX8FP#QAJ5A7_;jd4F+D(c3Dh7t4WznH"
        ";M0G`;fw&Iw+?LT{9fJLua(!mR}}MI1e;?EGbgw<#Y{J(I@F;M-dg@K&MbfVf1%2FUoM9E15ir?1QY-O00;m803iT%jzm!40001s"
        "0RRAL0001GcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO9aIW{z8VL35mV_`L8H8VIdIWc8nG&eUiWMX4sW;ir5Ibt_q"
        "F*Y<cF)(5`F*#*nV=!bfG&MG2F*jvoVPa-3YIARHb&pGG!!Qtr_krADP&X~4sh=XvRq7ICd4$!<l90xsA@uIGY=_c9R)h3E=9}ww"
        "1Hb}$$Bw|1{-QD9ix2JtB|R5xH|wp4c#>`yG$mUm6&G1n9Z=-5tm`5#TF%Q1TiKPBs(8s?^FvjyX{8OuDfgCM0M}(qf)~`Mz8;Oo"
        "5cmGFBxX@MOK9hP9V!rAcXAdV&%%uObZl%gJwUuz>XEbr67du2<v-wbhK1SNShjS2@9yoF!pPAGa^Wt53T}Xv6ZGI+pc_(S8=DAk"
        "E&mu-mcRVJP-(nxH`Dw9P)h>@6aWAK2mk;8Api}^5s}~k004^t003zK003opVQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gL"
        "b1!9NVmUc6H)1t4VPs@uG%_$TVq`fnV>dE0I50UlHZ?eAWnwaBVK_B7Fkxk5V>L50W;bMFVKgx|VmCN7WMnRCb8l{SkKIbcFcin{"
        "2YZJ@yiu6iu5*oh6)Ph>IiYcB5|UIL!*{o7J49huhvffpe&@1Y0kA;cu>&xrztRxs*$4MQm>x^Ene<jfI-=`)ow99@lFL`DN?!6U"
        "iK65$bybzBMxwe!-WK()sQ7M4D{Ux_xwrfRxXfb|yhvT{>&SQt@yVa(#3Z7#iR`pbk^&*+(OJ6x6lS1@Lu0e)4&vFSj*>=5BA#G7"
        "{|6e+urPTW%jVAS&8__?3=T#l7j7b`;Cfg%LHFK;bVX`xV>98k<sak1@|XV?Dv$T|YMeh%O9KQH000080000X09Mc2<wF7h02mJd"
        "04e|g0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-ia$#w1Uvh7Eb1rIgZ*FYu)y;0(Fbn|T`+&WJkX<&UPW5L>>M6VIDFOtPL^a+V"
        "$MDb4qS(7ncDtb~iU7F`IoJoo7HN{g2is4ps_6%80`04~x?`(UJq%+<#n55Z6nu5VRvopcSE1XsJAJ1xJO@4Yv=67_u^Ofv0bV98"
        "t&12*Y3?wB4<3`&DH?!etaU-c^k}Z*YdY?8|ACIw%muzx4=i9E#u|rHxA}8PcdQy%e>jbGWF6%>8a5n=^<&%C6-D+F>ycPnk3Bmz"
        "bi>BJYVPNh1Jx~co4F$n;V^ZHRZ6TV?sHANeNN~%?IZKAZgx#k6#s;G&3e6lle68jLfj~}o=)5Cu3?c9!x;B@jL-8_U+(hibbMm{"
        "W0OZ3s=x;9e=weVOo5jT&Dpuh+a<PUV%JO?sH;FtPtz8@wipgn$j!{~xA6W&DW?qGNWVUQcsfrkE+xiNYfM5a02x3aaRBXg)IygA"
        "l#5bRG)XMP6S24gVtI~Oyh98j+USB3R#-{Mg)NOzBAGyBe6XGy&6U*)@kA`IfLL8hEKHO<rHI_9EUFcihP)>%x#ke^7d%i@3K!yu"
        "SX}|JK1VFyA(kQK7gcB@yaHpAS6EtbDmZJTw?U(T<orTB6YDD=E-xkKApvr(EujdiETxbhKymURDxEYaaKh?xA)biKD<Fn*#OfVl"
        "lMJ9DauN}gPcAu9Qd)}n1@=~y(j+2f7UGE*u7KEFO6<XT<AsgNbF|I}An&p}y8MGSQCmY`9FGg}L~Q;C@o!K|0|XQR000O8001EX"
        "1iVedW&i*HYybcN8~^|Sb8=%ZVrgb?V_|G(Wo>O?FJE72ZfSI1UoLQY4a_kPgD?z6;T?%PSl)@~T8X6tJpk;H<@}+M2&tTu^EdPR"
        "^qBdTVo^EY$XQ6sRh#{^rE*bKiLspFNg)}uP20Lth8+kU226&pBqKwGn0_2HkHa{4f5U1&cux!Q>>h6J;SW$t0|XQR000O8001EX"
        "^CrSQlLY_(j1m9<8~^|Sb8=%ZVrgb?V_|G(Wo>O?FJE76VQFq(UoLQY#aLT!+cp$_50L*r=t&xEr6~H8!2_Dc>4LRMP^a5~ED#v!"
        "$l@ZA8c8K_yZ-kbl9H{Pm8`%pzu4m8xqas<d!Bchra9w4V!@S`Rjird1h|%)QZTMnNLBzO5wSWt`F#H2G<yH}{B#)>N$7dr;lben"
        "D@q|Xp>kbPsbG8XrxN_|AeEvZ3Dq>tsZyXwb6q7Y)-}DPI?LHzL%YI9BiYjKSSn(guc0Vth(@Mt-Q4pJ4+wsizh5msoqt?jTt^pg"
        "KQ33Z!SU*9`K~#>`gD1A{$YD-4LX9S5+p0o>H6@2<_v9y79}dEmh1^H1Y?F3B7t0mj5E!U`3to$)^;YKV#)3x>XOIgsF<dy$diWE"
        "L4NY#^4%}1894*0D+w1ue*&6pmQo67fI2)#ASDH5+{Z9H9Fv?WeS_h=C4Z9(fze&quYk;dA{f+T=cbL9L|PNO@U2`|#@j1Psqr-*"
        "=R#@BUU(<Y;`helZr<GGBEDBY5-td>v_Q;QOe{8-L61BaG*O}F#0Kt5Rd;BCGbN-+65)Y+Qa)<c3ojO^hbP{QNcgK_5|YKWtf1y+"
        "T$K=Ol!1&^K-o9mtM8zM4Dxc}tupaQ;tb+@o8zRE;sH42F%VQ_u+PE3;G(GTyXu==i?j-9jhgSx=evwSZE+@83~DjRP{cE(Rjvn="
        "n^(6%HwTA~c@1gq%3<r)uMLaNz_|y>oS_5kY3_pHk~s8wItE^B1?x=-3sa9flg*Rwxf|+e=dTJ)#d|>$*FxNn8E?<GCSNnYA!@_X"
        "Vv{NoQjlQckP7)oWpZ>!<9qXi2%c}YhFc+Bt6SmT69xZ=0ouoJZfC@!dF}<1G3jRPP&ukh-~qL?)p>DQ;j<T432n=_=d~=3DfX}z"
        "rnw_;(YkQA(<;wVU#T;q4Ss#B!mjj}<>(NdAu{fHQug=3pe6RM5ftqC*8NivTRo7NicRXCQ|2^=40|T_kB+?mt~LP+ry8%#hBRih"
        "1e(t?Dmf@MYIegsu)M^k$E>@mTa;rR5gFC4rP?neOxfA61!!};s~DDArm*g`WU<)w%+@uV{CJ4MyJOS0+`H$X%Tr3$DbVI)`bC&j"
        "MX5%J=JR#@*cN#!n8((tP>aZb<Oj0}HkY;$=1<CUP{=!>#3|x2_D+d6-Ol1S8^n#(T9ZS?zQAoTEemmO^GsjHeFE&%o^Zz+n82ig"
        "HMur$LfE${+`6kvgxSk?3im8HzR@2{hQlokDCwpaWe)<=(%dqEgSN8--A+lvGO?8pN%RP8ooN-Bl>-t~a7Gc&J<R(&XFS3DTD%@1"
        "_afbTgnX==`ZIf^tW44F-#q<Vq~RW|uU0jEb?VS-(-yGzy7kn+Y*T{M!cJIM&1OR|>TkaV&${8JchC5}>+o)z*XyjTc(gr?tT3~^"
        "c94J^+YWZi1sUlI{iWVa@Zqh-@*RlQcr(3*mu;Q-&os2z{&-si+E~^O91m3u0yTn%Et;9L-wo5?g<)y7g`VOXdM#v}^}*g7xq2^Y"
        "g{^L!qcO9xLxm9Xs)%%kgDz2ifv+QPW;Jo$BX*qn_i>*J@&6vtHdNmgPPg_vTV?hd<7@W=-{k6Wt<J9B**-CIix7@Qi4H_6)q}|O"
        "=p)cShvSgLy&;&F{v&jWrT7+S-wn(zIlR5Kc64w%S)$TqrPYA{StPgX*t1~PH*su*&5G8*`P=dY>ofb=9rq#Zk^P_%1rDp$?uMPX"
        ";!QSr-KS;+e2Q_gAAQruX-mNTX!%kvdGN%v|9VDJ@2kD>?GN4)=;O;_1y;_CT(0bo;6)P8xg9202p9rfm--5D=2nuESMHAp3_=u{"
        "`4UAo08wQAmWZMz0GMxBoU4t}P%MoO`>%y5iGKi4O9KQH000080000X09jot%})RT09XJ303rYY0CRF<FJftCZ)0I>XJu_|VJ~54"
        "Wo~pYUtei%X>?y-E^v7b$H57JFboCIyWkEXcU{1no0R5L3N|TeD!RSC+hgX_rV4}vH!kb&Efr``)j;tS)s(vZ8waV8quZ!q{!+Hq"
        "lFFduAUw!3AUiX!%Lh<P0|XQR000O8001EXV(>{Ma{&MVnF0U+9smFUb8=%ZVrgb?V_|G(Wo>O?FJWh8ZgekfZ*OoeaCvQ#O>5jR"
        "5Qgst|A)b+1iaq#>O)~^lio@Qy_Z>O<Q3HyGjikn`;4|5wrQiAKzio=m|++Wcb{hXMls?}3K{RbftquGf-F8QfXM=r$!y62n(A|c"
        "kaLk~7>2&<JI6V~(YtIgEDbc7oGR;60Ut}|3Qttf^Zkag1R5{bM<(mfypI@S(77D!8{&bmu1w!^eIm3k_sp5w^u99zMc^y_rjQHq"
        "VWCv#z5_j$9PpZMgn}D^{>-2Ml@oFMDV2)M;8VSY-RJ)lUU==O>xZ>TnKc%)dm&_V+Uv`@uO#)7k&NK{rjZ?2yO2iFa%{oSR`sJ}"
        "lGD`Wzj1%G&=*EE&m&YHDX*7r&zXNoK|?z+YSF3_GV8gCxa!0swUh!ybxwA=G7lku_)X_Vt<#Ijf>j9Wi9j$CU<mnHfvxfP5;Zb9"
        "8=2l&zg`%<l0K^CrY+Sl!tf=hq!7?Xh_S)O>uwQ`fB9&m+s=@#)otIs0#Hi>1QY-O00;m803iScl}$BB0000=0000Z0001Ua$_%I"
        "X=ZO@VQgn*ZEayMVRLh3baO9XUukY>bYEXCaCsHU(FuSs3<bb{!5tnMzy(~w6#@+!Y?G1(q)sp1?-9|VW>U~3xo|Sv>fB+D9+JWO"
        "=OcPvIpBfxTTEXGbe`3@jE#ua<pWSl0|XQR000O8001EXmZN1%;{pHxVF~~MCjbBdb8=%ZVrgb?V_|G(Wo>O?FJW_YWpr~dXJu_|"
        "VKiT5GGZ=jb8l|#R!wi)Fbus9*ncQ;ZW7yx({#rH`xV0=&=&1zl_gK2-Dbu9`zU|aak}2N!_XiI0=_4b7UlQ!{PDxh&8o5zwNP^P"
        ">1I`0TWZ{vSXHpzufHUp*NRrR9nagw<tQ4b1M@^w`%l~LsVPfk$^vrS<aYaNf~`Yub&J~yl-cIi6sp{O_tp%VTXd>WC<RA2brL0q"
        "bsVjubif_9%BvpSVUp|*adAlTcps;`c)u@>AhWv&ciAyd^8Eg|PvLG@+SHyLLM6z7@r892Ixo23m@nO=n`k4kjfNG3PbB&t8GjBj"
        "jsg)H;x!^u;29`Dd%V3urL(ndhzZdqx`rPF{t)eQd1k#J`+%mOLNdlBooftDW`(vu@nJP0Xdu%lDvKi1Hh)-V#LJBF6o0DGkx+(^"
        "cS0&46kh!dMNFghNGp^9Ewn8a32zI^p{{wz%-YQ!tJF4Q+6h4lsVYX(c<Ox?P13^@RZ?<OkdtbGh7pvg8&RLhy8I*v%xSft4;}Kf"
        "mx%%*_>qK$Gyf`6cY8mpsEi^7nt4v}5h$@AP4a?YZ^eNR)rj|r!yv-czxL7aHfSYzOT;wWvFX2r6+E7QR_wiz_eTE5M*8@-@!hWA"
        "Nz~S>PMWdGP=-X|J&$*BbT?^4?~tPyciy%a!6zo6GEMbTrc=a_<v53X>Gcbx4alIQ4|o<W=mw|tOX47Uog@W(m(J1a_3;JV3V#B}"
        "E<3GHCMmC3tY>~P#a-ovVbc^?cL{MFP0Gqn$Jz@HDx^N&^Pl#826e?;kYwR3fjbMnI<WM2vYe#1GhSDKPDl_+N8M*~pBZDAOC7A&"
        "g?gY@Bk9*$=j5`j>vC@tlUP8b>7`EpXDl}rdsIf+$4hG@h2qgVd3of>HIsdlt;1E})f3Zr_Iyt6Fz=uB;(?aT9P$$`vWK{|O}Icj"
        "D#^D9T`w4|%^@4zFgYYMKYW2+^WBabn7kkS{S8n{0|XQR000O8001EXxU6!B;{pHxCkp@oCjbBdb8=%ZVrgb?V_|G(Wo>O?FJW_Y"
        "Wpr~dXJu_|VKiT5G-57lb8l|#R?BYNFc7>K=syHL$F`(MPMTwZenn9bD{&>Up-7da<hViqy-Ph9S$1<P+Cz-M2zF*4<kCC)^6|sX"
        "&8E>DB@=w}c(bXrt|aa&Y#P|6+b`+ot#F&$foFAR3S^y;o>@EB+Q<FA?y5?ts)Pb}h2CEcus3M6>~Y_KQ2RowQdGO|R;wv;k4BUN"
        "IU@^&PNL+njiYUpO}N2cSTTZ#5tfH2xjQ6rQI=Vh^84%|IV3#AD1G>pK?Y?C$>iF#))|C`IRna;+B9e^ql9z3Y?tk#9mhVJJRCkd"
        "Hr7b^i?@*#oI~Zvhvfn+0VPO_w-+cky49T{LbQu6;U|VaSig49wPj8}p&Gl8%y#ZhE&3w!QfhGVVKXDBAeYD*jVzbCcv?rq^N4Y<"
        "KeuQc6Urgo1Q+0tS@F{+QJs`CtrQ9*le!X)SzWpU+LoG3th>drDyeg#)d-w%(GZ%&q4tF}$qvCPIH#hZaH0n{wV*=TvG(M&DQf0G"
        ")teQ4s*vJDR}e-=ViHdDEmS+){UW1K!U>SXGmM@<g=1@ySN!rS;wVW&rZKYbgm3%T(dvH=QgFHx7MeR({SUB`diP%x@?PY7k?%$R"
        "-6BW(w|PD_@XT6m#h@?P<j8#>Kda)qIJysmZj||1Q3iodnHRe-ZB^I2rWwY%kMgjDNByuXx$wv&22XfmJ;)A&|C%^W?qRHYJWZD9"
        "rT;nHGh2h9RU!Dtp}>|DXCZoFf}6%NMas{x9UO2Q1$iL3CIJBh4Z1Pj%b)Ij0S(EF*A-^819M`u|B!wS+LC0q3*L~x)#zXlmf;g5"
        "bitVV()#_u_<SHYGwIh`@8rC!%XDuf<5)r`-D{rlpE+Gfb`*;1<GH>&f$XRaKYw!0bYTZ^y7l|c)fdxQT3jf&!D2k?*@4!~0*V?}"
        "(Y>AP&Tm0S!D$te?Rp97bpd%;9%x@o(@W^u_ApRAlP-k*egjZT0|XQR000O8001EXgY6oG9RmOWFA4wvF8}}lb8=%ZVrgb?V_|G("
        "Wo>O?FJW_YWpr~dbZ={AZfSaDa$jv>ZfRy^b963hb8l|#mfvbzM-0aA19=DI>p0cF(TsW-=yL?kjO1==dxO37q9N~IEoteJ)+L2r"
        "gf{GfUCx{%X+Hh*&3<_K;_B-9_ISL#i|=mZ{@!oo<_CIn{r%O4^HME7#?ObR!||bRU2n2XozfBpx>y8tW*^Pg`9s^<M%TQ4_1nvN"
        "xQnMhg4LbV@uzq={1T6^-#vW_iZp%MpxKR=$OPVvoEo?#Y?!J#kheG0X$XtNbvvlF+=!vSC%X*(b{!w`aC?!XHwhM@kiO7ywYFQU"
        ";Mj9X_aN#quxV_ZxC8bO8fdNSm~CC;_$+d?^r$*(-y=!5rvgGJMA12#Fo(NAXt#Q->K(wVXGXo&d5(+7xA72<ek!d$FMkJWI_};("
        "kG@-m?rh@Lv~63ENXXeIW0rC&MFzC$jSj8lep&we$K#5pr{kX_VwyXpZSpN^kvR;aUHfX?anO#gDv}^*u0*32P^?i&X4Ri{{VB!A"
        "^Vi2$f60|R$CV(uW1n5D#A>t?x!I<@!aEywZz9|LtK&-DcraQ|0NB)tbx7nULdZ6gYEz|Uo}0o79;K!b;4I`46rSfQsXNr5F)g;7"
        "wze@4h!O1238sZ7tOT>_)u`VFm<8491CrqX#1*}{Ih>wBi)*q)F}1}oY>bs__?{i2rf<<IM5KBDM`*1zHao0f71gy?`OBDTrD3-0"
        "xF>eBYMPf>V(n6;Bv@0a&kV>fLCaDymXsMSjkR}<tj17O(C}s**V$BdNd%Uh2C7aoVk}lt=1a7G^wZl@+`qr`Q@nn6czEaM3{Wi{"
        "(0PqMyz)4JQ~)bSHyaAtxX)Qd11ckMqt$7&#pPqG&GtOJIB6vBLKgv2S2HR9ode4m%U0*^vmo%EGHPZ@rBT_V20m7e6MrGRLZovB"
        "BlfZiYCK&OtP@Nhv~<Y|DapOS@_<%|*KpWiU*!eymB1^118)RGohRigge$Xw?B0qd+7ckcnl~>h=o`HMBD~DAj1}q-;zcnUYk4;n"
        "&9)nt_U=w8bwunLrBj79?wm6WsQ&-Re-qx9FY5XgP)h>@6aWAK2mk;8Appu<X{n9@000C7001EX0047xV=rQ9W^ZF*Y-eR{ZDB8C"
        "Xk}w-aBpdDbZKs9E^v8`l0i@7Fc5|BmHHn>Ie`?~-D?heKxz>Zum~isXlhTI5wRVOCtWSSe#c3H-7N=lNjx6=z4>0Ms(SqPy@A}J"
        "kcR;U11Sk)62_iiQ8`7x=%F3^9yqf8RUafBC}MQ9s;X+aST0&g!LHG+=N)huk|^BU@_lc|ZF1P}DFj@F<f+f|>#h58Zn!E9w`JqA"
        "gV|x*rry6D+7jw0beHsj(Er|uBq@^1g{KxW9_ZP1<X(ovQO_p5=2WSMH0m(wvu9~0+vjlc1MX6!X1M@+Y|8Celp1g;w!B~GG>Rh-"
        "I&d~5d9CS%BEK5L4s%Dj2J|=>(>!4`c$mQ;TUBw!R&EyLOrOqiRwAlusIJ4%Q=q8xB^Mc6f~=@iCgixE1boRaSWsEnnXfzsOXvK9"
        "hLtqQKR`Y{{~P@T(J2R^-N=>%itAX(mFi?fnKE2%wm%;>kOt*YpsZ)Vvxn&?`VxY<VNyQAcrpfl8ok4T>h}O@_?6^{nP%i;G%FtN"
        "Z_G!u#y+869W{;1A5cpJ1QY-O00;m803iT}!ltQw1pojI4gdfe0001Ua$_%IX=ZO@VQgn*ZEayMV{dL|X=g5Qd5u?FZ`(E$emCI%"
        "!0O4~Y}b+{--ZlpQn$;{ra_zx1A;)H#Uq=GE}lf^(#-#ThZH4Kirck$sL6A9F5kJzvaGYAJtqN8(u4;zA~}ER$0>t|WU%JB$esse"
        "2|!pH<Zs_yo&A1(y*RtRxmpZjMrgvw>CO44_x{J5^Y`*<U|CkX)o$^4n`VOiou^5=6=rEn7!@>#Xr4n(G?nIz2cmZ%Y1S<jn~1MU"
        "wo5!}ig@1gWL*+IaRC{Pq`uG6<R$+K<fOd5wHD5))g#uzJ+<y6Q@izBrK*GSzCv<xo$a98ZmBzRN`w1Vn!sh2hCG6KyG8I{j$-Ge"
        "(okG(IAf6b1&t@7xQ*yDWVvy1|KM+cvc^@up&2m$759qf;Ail_bDk!1g6J3bTpYXoX9wA{As$_yYBN#{rK0$Z$0gD6#szsDrK=-!"
        "&vvxK5oZXq|Hu=TKE5LB=Z5u2ofNbnO$0m%KjJYLCCwh$!-;<UfTpc>5k<=@?4)gD$9CO<Wg7viW@7WoaF*^?1rhu>+Rn);O(SIZ"
        "K%*Rr2wz@(`EdSev78f`Tenrzdd=Y8;k%bYtV)!5lc7@(D~8_TY<r`zaU6Lj&G_vMjf2IUSs0F2t``ocHg#tJ&SW&AQ*Rc?KZngc"
        "X47$KYO+S)IPTaE#~w^)s25QRr<AQGLl~{5ffEF?sq4{+`KSSt%5hEZ_DYxC|8v=`xr{L|opBwLX1YH$t#@skQhTzRxHC#;p2I?W"
        "<jt7F0-%%8%$`usb=i#8Tm~3II37A9%!wcjy`jq{-V`&0xiEC6Gmm;Sbg5$*KJ9;~zRxxg+;3CNsw)U}b*pCQb7mCQky<<21F{r*"
        "#G&0l5+FVEWfjZ5!=N~G@q_JW@wA1E$B>BS76NTwLSrxt(MslZU&z1Y0>h}Q1dg#3G%5s&N8fzuw_$t?*&3KMJD|IqM$%;YKydI&"
        "$4ZAD(@eG;S$vu)qdbuvBZGi%O|gvQmGJWrF~EW|&)Qg=F+7T>P<YqtSphFwrN5OxH;cXSH6$YSWvzE|h{7)E|Eel~en_Yc#4bzp"
        "v?;iXaLC7Ka@c7^w?}^J9e|c<mDpq1vn>h-%_<wutrT`wCX&)zduY^HXsy$99l_f*#4+u={TKOmZ;_u#$Fc|yOCt!N%f#wqCUuQU"
        "(aHz4vGM9z$z==?Lx$8?D?T%cUz=d$bXk1{mRriRP6eESXO)uGokH|z!fVj#v>IxsJd^|OqeAUP>)ZEVw3fZuO;mg}OBK|j?3!Hq"
        "`GHGin#opG$>FMm=j5bxvX0B*ugk^N`RB#OwZFVvoGmd^v@Zo()slk$m8wdLTdglmit|m1`$}2^Nb93cHstA_Un95d9K8YpEx7?e"
        ")f0N82FW)iyT>HEKPb6d1MTA7=Y<4K?63!Pa?~@B9g&1-M&t=s1k%LVNXaz_A1X6DRWto|`h8EZ(NwT94#r^cmE;52NMy;TGn;a("
        "p+9M~gGH95Stqn^lKUimOvu4T@*+jO$@T;68)=FU)aLur#AH9P3_l+z?rc<m;jAB1k@2UF+@_V!I3(mMtz1`d-qr!=k-=bar=7q("
        "UbQVEHb4UT3s@f^`D6Lzg3vYg`&=km`I+zwN#FqsAp{X@iWjQo<;5p~>R0qRN-4u=%I|`K{GE5CNy^6VAd?HJTpx>q#es5S?L^q0"
        "THUUmerRwYfE{)>WowS=V(O!M+1+rjF_$o_F`bZXj-lFyW(*@_Qz=<%jul=xl5bT4l+&mC-ORbuV)lBW4=XU<oK#+!Gpm^@)*)7;"
        "2rQ1dFhq}w{7_-4zTyi1`pXa3htD{xTko$83)`*#08mQ<1QY-O00;m803iS(=l*9e0{{SX2><{c0001Ua$_%IX=ZO@VQgn*ZEayM"
        "V{dMBa$#e1b1rasomOpY+b|IR9>{+X{^UUGwfmUS(T-+N*xIm$F<6ID>~kWb7f(8OZP}0CNwyQaNz-(O5Og{{clVs`v7;!uJUJaR"
        "p%HAsGjMDU_y<6<%xH(U%1{{1=F(&<m#|v^mYdAIyS$iPq|>YE<jd9dBvE-1MbU6D92C|lmUHBp<jxVCP^GcBF&(i&fXo}ulH*c}"
        "dFwT;EKH3P4^y&xxW<mgW??-XoUf5$LU;mwHnSCshJ#>Z-=L)P<p20tNEi<Xj2<6l%&3}rNtGdJY%H}chk~UkoyAmWfoU2$kj1D0"
        "XhGOnKgHmb_?^+lQ53;95iaF(XJmyCkC?G6N|K0}u6Ijr(FN<>U9hg9f^popb@K3!<-!5`&SeD?YmJSgP2B<BF;zLFUNQ@kqqqk)"
        "y}7;m(QFFsL=Fs>z?|l#TN+HF(HjZAZB911S=#83Jm(J5%qS(WV=bsDZp^SMCH(C7>WC#t^6StXxUiE4uF7?fsU+k&gH+_5c+ONd"
        "gRY^Jy@cBG$2$sZfSs$l2*~XMFeM_h`ckpK?AmCcyW#&1Dqv<-pv4p9yDFhcnp!Bt1BGT>THyI3yP6#r#wv~o`G+QlaFIC`9vYcR"
        "N?$M9J@g^kseG9r(3#1>l5yP@(t@@Z^<>l@WyoodV?)f#4hoQaUdAjJ84_PX31S*OUP(+zemwhUesk+Cw#?na@Avfk6xe@Of-b8z"
        "yw{6l2HKftb$SA|m$X!Ix7y{|*1H$3b$T66@;S&YjE(4^2_kYti+j;rRk~5Ndu<!B<Bu%rRvvZqzyelQ_lSx6Y7taqi=~7jH~2?;"
        "{QDKYa%hXY1O49=q-J!t)2OBPytJkSi;o+o+ls=N`&g7;t@57VZhyAfjrVaPjb6l~B&(FGRD66&c<hwclBPOULdPfj(pt~gf2Z{V"
        "iwCB)Ys$V7p1i4$y4VDjW+8CrD3I(tvp0hH1}P`K^HWc$?^Sc<(@Hg0T|M3i@j(G6_w?4}K9m=H+txRy-Yylls}8cQbU(fYdY#`x"
        "9=?}qljmHO*s}>;SYCfS(Z>SmnvnGG8jpWaO9KQH000080000X096G%QiA{h0I~o803iSX0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-F"
        "UukY>bYEXCaCudYK@P$&3<U3$`iGSt&<`L^{D3P$kx5*L<iwTZ3%{pDT)6J+Xo%=4n~fUgZxjwjwjRzo2Dle+;Vs0+2`;SP$B@7("
        "kzchxQ3tvOYTHzr^pLU;(LBx5>ele6t7X=ww8qOJFWFRiuN{hcV9#^)%Wplzlvp(-oN=RnO80sC08mQ<1QY-O00;m803iTpza8qk"
        "0ssK11pojc0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FJWY1aCBvIb1rasjZ{r*8!-^Q2l5|=?Zp9mH;2%JOADrUiXlx+6WT*5VJwZj"
        "B3emE+Bl~4zjw4=e}zyYW7hJ{$D8+_R20SZ@@fny(I*NK0tshh3av=s1C5k^hG@~p+ND^+9T7+;Vl|!IPaYQY#q4H&J1ZMm7DaJ>"
        "c7CRUYd|<-NF>dbN?Z!ijdvlzoI7B0p^x&IYNMCCw+r6xD=f8g0?mq=21`bi0*VxOq2I})4zp)$yrI`JqClmbk-LOz!qf)3b?G~y"
        "+{{AADDp=bj1i`oFjAUqSs0(60k7!VAP7D}(OM4DD@(GQF~5|gKy&N^ZBfKQB%_AVpw)_E8h7&yA$p^E!;D_Q=+mBpUne8{?mOni"
        "JYKgM>=?YZ+;WG5vjtjbHBWiUz!26t)!?2e81#Dx<qm6JQwP>1NP(GPeVRrmXedTpQ3Q;@xms!Nwrp1sT!1B6Q8ySKAz@fi%29`M"
        "tcSMRu0UJgW@Ai5Y8jWJBx?S@I`R?Qpi<iK<XuASfUI>OS)Po2*l_VkuZlJ}1a(<#-eL%<m}eduvI+iV9m*|de<6yR8e~em&&|=w"
        "y_?|_VtLg6;@BH^KWBH$R^6LpaO#zdUb=U1lkEqR=h}{3W31r`jh2}AdGP7wdnOFxnqsb4`pt(uV+^NZPIJ&z`O<)lVblpy<zHQL"
        "ag6bEC+Gsj<MsXjaUN<e^-4HN)z%E2b5$x{4E|-)ytP~g#^F`}JgGF3Zv)3B_@<Q6mVgiMLq7K`X{_`?rhE*Rq#R#O$&FBPcD_hN"
        "4biZs17`1HZ~Wf$Et{#YRavQ(%7)(>cMPefO+Hxs4PleF>=ITelyK{So=LR1?{Hf&&>Dlx=i|rMtf)u6U7X@H-muWPyGct}1SfmU"
        "JNt1=Oxl<ujr(i!luO(CV5$Ra9P$rPO9KQH000080000X0H)DR!O#H!0BHjN02}}S0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-Hb963n"
        "d396Ej@vK{yc_U85I*e!iMN+tQnYA`ePAzXk>pk^0!E^3wX!5gbg|j5FBL!9Luwm_Obutq5nWZ){k!)YXb5?T1oX-g6&vrsB=8}x"
        "(8A!@SU~8JRYw}|WN`**YK$XYS5;LOWzj_6gGNSWF(pc1`rsq~4K1LPX}m()Qr}~(yt^nb_#FkG4EuQAI}EAwd@U_0Y#ML%ob@>N"
        "vGu$o)aQp7eawPo@p&oWMrhv1eLTd0Zp-2lgM<G??#uo#5u{C7fLzXNI`TP40C)(Nei+9qsO$P=1ZWD}UE9pJSTj3%<ScN+#K+0&"
        "bTcxfslCmWmRI1TX^kAgl<%7+5iffGsLJbQ*1m=Hr~h>%$MiVlzy=PHXf1+F5l&|&i0E25F-e|f;Ua6^1fe?2jmT;$qRuU5${h)x"
        "1Mg^=qaZMHO5BShh193)pp9$AkzkE73>`XcbHq3B_1pd~psq*8hgXw0?@0Ym5kvPq`k$o48jwLrLoZc6-=_~CEHZ-5GoYKT7LYt`"
        "PJ^fIwqE)k7ZFXOtAjEf$OSEbMYV$Jxfg?batv0y8L*|h-xQOK{yG>ZV-Er4Q!15SCSw4tX<bgLCrO#IKTt~p1QY-O00;m803iV5"
        ")`me10{{TW2><{g0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FJo_PY+-b7axQRrl~v1b+b|Hk8}L6YbdnNDwR1F@0%`1nfuabSpclgs"
        "XldmTp-7daTRUH0QV;9tJOl`k#M$A@a(Cr8&T@9M0EVd!a*`yN_>dDFKL>)d6KEvyGfGH>j49uT@Q4VBh}D<n-P2w4uzpzmSpQsw"
        "SrR&qGo4H)G|Pq5piRKDeNUyxU`JCC6Sl`JBVjDKmL%2+D(oZvxiR;5yR8pT;srT<#pD;}gv!;CWI01iHJ!waC<W!hy<m)(Z5GoB"
        "m@B~)qKI;;qsUdrQtKm$v}KEEvH;38+``NUy5~ZofhKsYKL_*K%&VB}WeLJ|yuq|zd&oAWLAGD5>_(qS0z2!&(wZ{qr%G?ejc+{&"
        "zI9U6l1WqGLN`RsioGPE3gI_l2V6-hr0d)hZt-!EIt0Q2vs|BGXL&=%YvPf|B0(AD5|w#I<7%@Lg1OaRVGW>(@@#q3B1}ThdbS;<"
        "grD3#t!KU_g$I+E;p)4FmZAvG=+Pv&2rlc^vQ}(CwQyEt6$*|crZ3p&spuv>tmv)ilpbw9z|gc2n#etByV7I2;HlDE5*AiU2FMP|"
        "u$qnp+e5@iEb38pOjHsx%3b$r<g|Rax-7l5*r@hA_zJTRX<zgFpi5P-UaC^@4ox~Z$34;XFZRnUVg`E)89BP<=~MO9A~IH5L|v(E"
        "+M#6+t~-d9ZwO$1Y42u>mI`Is*GOp5sbNnUKrtP|^k8b<wK0C32+tpm9v>2Y%JH@gVuJ6xzAtiCIr%iF)shkQawdI6DOAMh3)U^#"
        "jkklLcNUl8E4^>VGqOXbI+l9DgQ44NVk!RlokQb?45b-)&9U^#x$?CsB7=*CxY(X$Y14S7<i}tG>LMK<;(n3`?1sl8+fk185A?>!"
        "xgMD|y_sNJ(`L0M+$@6mwu>oaEy(_n+%s~%*m|SL&EnneGCgFCZ3btyg)4A}L*)Z}-JqVbDyEg~()rloRVxF6{&{s6hDTHuYztpy"
        "XKc%QsbP?)50ZC&@lBi6#p6TMIaq%x_cc`i0Z>Z=1QY-O00;m803iTANXHGx1^@sN5&!@o0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+"
        "FJo_RbY*ySE^v9BSX*=BG!%Z%uMplBc{g>_4uvHU1_sy}USI|on4z2TD7KtvYs*HGyUi~B?|i!XlFgQZOxxHxN9TO!s-q@J@*7io"
        "ZZ0atIVri}su5CXBMMUU2T`z^6s<J;!H~B?mhD?j^noc}lHHN8x+YpYkftqpt+OOa77NjItuo}m^r05J`1(e-GCnuV9OAR$i%PW("
        "bksF3jA*4Mf%IM5L$C7mk{MRiOlv;SnJz_PmLqB5S$Cpp0ejDzhGiDJVucxY{ldFe3)8COuab58(3-_!@y#HELc}{RH@~T#rwg|P"
        "AkO|jocZeo!KZ@>Yod*EZ`v?r@>)o~CPEsw#$|c3;@l|BWLwcH(5t#-7DUM_l3V|4YSlTulw5pGYJ}q!FI%r$anmb_udD(hgn+8#"
        "mR-F2_TtZc8y{8|Uu>TzOJXHub^Depnx;XC0yeI4A5QAO)^jVJUZ|XDr0&Ijs^r<TXXKTr0sp-hkXe^J5^8PM!{X?f$BVcYJ@c5p"
        "mT@ik<}q<9B2dk3PULC`Ssqf+Bhev%>oY-JIu4<A&z&RvyJ_#Z6z_O=eeS;+_K-V@cEKVW^6Bb}&pgfJb9e{p%hk9q)I=m*T`w2T"
        "cc$-ZZi9yWM3#oop+J6?^^tV_u10}5c$Yg~ArD-nutHc-@Xqk^OR{h4(q)>y#NXmTP<BM}d#>=b?;AMG*PJu8A}Cxs+OnKsUry1c"
        "<STMTT18}QJO~6ziH@f0A;gLa&B^bq?)eW&wTdP}-m_YiF$TGh4k{K;#t6CBN$R%$)pK=(GeQB=kkn35>RAdVOM#+8s%*&Z)`ydG"
        "i8lzNo(XOt6><!|B^tHMOBzF-*>xCo$B&zuHM^3L3YFxNaAE|UVj11ClL8eZ^}LxSB8RqTcpjDIc6+cb0?k-i(n0n}dq%!*om+j8"
        "k%K5pm<2<PzJ%v0vB&-b6{T$e_FQrWT_rqH9MFJ;f=^o|xXuQd9csw>X;2pnI@5q^XmU9L2a}mR(wUK+uS5M-m;+6+B(<tL>yY<}"
        "Lxx7-9_+V7FVE4?dpOv>-@}krC66u~Zj-|^C`jiHwb>zL&!>Q_KSuD7=aB$PceE<9H%?RLfn-hT-`b{j4@I^23lAnq|4p6mKfzg{"
        "t$05<3ccgJvj{?mA37S0b561bDkk@wu9i3frEh5IyqZoxxYFIAy0bT}2s>`Qv~`gBVzz&q&W#wVA!D6IN5|A!4g#*0uENjBn<7np"
        "(+`CfygzDqs={SE6nE$pA+c-$gM@N+^?Fj+uDV{3<h4xj<(2zvarkrh<J+5@An{d?xdZ(|7Q#g>O16TX2&dHUn?kSz$6@8!C6AJ!"
        "GqOF;;fp;Gp_TLvwZD!hySJ8V8nB%coRUlD$k`Hjuly;<<Q)oi=vP*XFuYlk01Lw2p^1(bE~mn6AQg@RpLk;!W@tU@2zBf^c2)q>"
        "DqGER)+$)~j++lAx!25M)AS@&t-B${>$$bNw==pZE07!CWcKqmVn0`iFtt7!t;BgZjiMI{wpDgx9ZEk&o5`NPd23&6$cF$XrhY(s"
        "ZsDI<wFu5EUuV}JtOmTc+_G!%&{cMQ;{bntgVzwO%adG7KV}Td&xYZ4wT(U7#>XGIVPSpV@|#JZedfj9FdNk9bDx54Uw{$V?U6|*"
        "2P9@ZdV!J5a-yAu?lPUx;^7?T{Y{wjezb5gt2x+@!?@upB@R^R>IDuC8ul&;&c|GLyb!Do1Bp-klzcrMxSpCfth$^(6r7iOXiU4l"
        "#C0X~B|m~pi&EbSc*!oTh9dR7EHrj_W>h*Pu`3C0wQFR)m7N=uXR$r+f++OAJ7H=8a^ULeqb^z>6XO)%7{>w9DflQNC7aD`o}Fpl"
        "DbDVu$4T7wuZKy4DHxMfIIAjQJ`%`M>@j`-X}C6KxMN=5d=}Fwx;E)eb@38IKz85{eovSc!xd#nzxM;Oq-!zH9g{Vl+}NG_#M(92"
        "H__bu{wi6HJM7?{_nMd3^orG*PiSM<K%7k5BL{Aq{Sa?3{m(j?NSq$^5I8&e=g8u*y`Bbh!V{7!Z_&PD1lBR-CVaz;@4^5U@9YmS"
        "SxvY8A|BI^{EewH3|^rA)Z(UhQid^owqD+2xb7iKTWSB}Fize!9q3c{y!l17V@F)3T|1yvXmC}W0{yvP{ny*XcG~aWyIE&4+xU6d"
        "X?;w|CmY8V7M{FKJ$=mx>3mWRgE;%&l&tKVPDfo!g3Rd^&zUZS*!Zl4qcN*9_)BkSvb00=+7?g9L9c3FOaA~+O9KQH000080000X"
        "0Dkycq&5fu0KXgn03-ka0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-KX<=t>Z*z2MV{<NWdF@$CbK5o$zGo)?0i_Qi6Nz@*PN!AtW;$uw"
        ">7_G$UStjfk&uO&1XuvH<+%Ru-37p>C|T~~R?Z|A$=$_1KkR}*5d6ZnW-di>M--zuQzYl~N=R*ZroSLGr<GwjsgyJ_lf{e(b^}Cm"
        "z2+IG1z9nXZAz-HiO!^AIu3&1`LpNGcv(qhKqlTif0nZ~$Y2#!EQLSv51JaPuXUJH_{+?U7<tVE|G?CYyquAeZqrN(1InfLjlTHg"
        "?DJ2f#q(zbuBu)YJez??enSl-7i5{t;GcWDmWtpagbU*3FN5mNY@`ShFTqNo%4o~mJp;pl9G~7lbWc%YMa^bLtWv_DXblFkJj9ij"
        "fqxz(F*mH#VHEWi!fWHadqLL0r8qe`A>Z<XEyz8JeYh0CklL%U#J34B>ADOHCc-Lmy5Ra&{fN9G_`->ghEgm=3_fHc4?Rh1LewK("
        "*F^!V1VQ|}<gkjBJT#w+oNY1uEH4-?&qQ4^1>W}_aBp;wj4ju)RvE@<<f*0OfY5-EL6XFfN-}3*P@DDqvtVS)ih&j=#fe)kNW-oT"
        "sGF|t3_~F~J0L`{HB+!mPtKGS(g_F89VgoIgV-i=BjbFSoE0NhzP15aIuQ#m$}zMWM+y1pqBCIRV{07cnvvgVQL}eSNfib^fy00N"
        "x(R@;s30ZRB{kV59_!DAHmc4HmjbK^cg2G=qc~JQp(4jvGHQs~Fv7MpGhl&m%sJDU;wu1+6)WWJ3`k9_1Q%Dt!VHJ%LO;ekNsrg7"
        "*7HtIg))O=NEq!YtpMxlC-z6pv;pb+gu!#D!JXwj-jJS`cj7eR7mQL1UQG^rm0mSMXMm)6sw;+|*&y04s%BBNEAgud_!~LkDdEp#"
        "SrrUW($_BAX7EG9-wHsKo7V7EonJAtn;B~&r0JXzL!$61e3A~VtjwKPU}u4XtxXC!#BVH0@T*Pl7M-+9jyQ@Xv#_<2#2QR71s6<*"
        "*9=lq<leob`bXIf?{t}w#nIAE$UE=C8A0kfE%=py#0Q7A#q<TijSgx?-$Fug4TXmA2XAPB)gT)8$3x+Wxj6?N5v_nqQE|mWNDR9V"
        "pbw&%Qyp!JcJ><|xxR0)Zd;LGl2*<HCtexQ^W1H(NVlB7Ci7N6(k1sa`*gK*K|EMB&C*Y*o96oR;?V+Xu6?#!`&&F{)tM${g`UFW"
        "&eVQ-e@lUSm-v8;qxiwwgTk4JwM&#Eg54PY%86XIA^$(n2LnAquBS)(H~}7y$CYg{x+h@I8Y^h?uP24>3F5!Mzv1=OROkR+K0e>2"
        "-xs?*rks$UjjV|8P6UIQL&N4KYmrlEU|_0oGtBRGSrKWcnUWivaxCP>lg?>t2^2p+J3BjGC7vVPueopj-5j!5+}m#W1^)jcd4PN0"
        "s=N@iWbkCvoVbZ6rWNi+6Op=~)+fNW4)soX9xQsg%=XZ`UJ*zGc0-G*sP(Cv*Mh?wFxA~~jqMe%o}Zn6GC%ute*W=ZE+vb0mAyo`"
        "bEYNa8&Wn6Cah%p&;xwEUKWk+?)EYGGz8+E#6HZCRuaN;hx++ZIsJ~6B~4u~l@@HvGkK+Gwc%Nc`IX+nBqeWkdh>d({>^lqHe)x)"
        "w_q^A@GyKdo&eo|{NLxz-wjpUcmZGg`)D)WXfZwx*gbN{l++r5%bz+Z2*AI8<REWtqXx2j=wO$|wpQN8kP%^EAFN^cZ&nLt<o+RY"
        "Fa3ScZ;^Z3la*zs7aVRtB+Qoz+I!QVkZjjCf!U8k_2lF+Lb}}+|1E0u7FKu%fd&s$wXv|ehUpKmRuT=y3P588X+beB1{1&lAE8o1"
        "YgWk&#j~3rS-RYUJWP6AhwyMBr=OC~Jq)BJ{UF6$XB^Y=G|Tqo@)QbY(BLzbR$RbLkYXw-nK7;52E#uo<}p|t)|tf<Ki=V@Qw&2a"
        "4Bj@wpxYa_#?In+<X~8dQKD-Dk-)dQu3);chktC%w=qoCw%Y6zj|7+}T+}wCGjhveW2=on8NhKLX*}<wY8~qC2ipmt1(9fTjbJM*"
        "$avW3;0B(jUAj#!ek{iJfTy2+hj8_u=HHWJJOELwo-2jj4tN01o*p&mF%%V2l`z_V0DoH`!FV*V=#j||u>M@YBceZMQ!K`5+a5ZF"
        "DtQ0?efVbKnxH?cyFab0H<$Y5r3mAfZ=y>96o7yZset)sFAopNMh9PhZTmPzldJIc&?E#(cch7KjFs2W!ZkiYyzZ;spZu|NZ9HJl"
        "w$t9pc!>C^(N`?zwzzcT5f8ujYB>mc;`Ru8aE!bU5uWEErM>O<xVsShO^xRaxb1X0*mDKlLh0kn_H^wQpD^GFD)hKpMc5YX!EXA3"
        "Jn93-vATnoQ{<4%<85);oYn;zsI9mNW{F0VgLeHYu7$i6(^ir1zv&EvF`_Qq2ZK)FsMbeyJ~n7niU?&I&%$>p(L!s;n8lv9rna08"
        "u?@N0d^oIPOST*fNFs7^u|o?~9mXV4Pp8pGk}kIHc`_)+p0jX+S3K)C&dxI1r8)C@%XDzvesD@{QBI(CKw+3GQ6#+V3an3I<=gl8"
        "VFj$~d$pIPKI8-59%+i7KaLhcPJ7zeWW`_hJb?Tl#lVD;YYkgy+Dmf3jve!tZRR|ptbb;kYx}GJvfwcH(nfgjgCxxdgF9j9cCO)l"
        "0VA`)?H>3yvdCv-*zUEB!znb{nyFLRs%#(8q>z<Q@B>oU=egMHnEVQ+*A;*-qoo5`<-Tv>R-_?9bNkR9VS^&t;*2)|GVd=?O9KQH"
        "000080000X0PJ0WMo$O;0Aw2g03-ka0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-VW@c<@Ze?F{W^^uadA%50Z}YbGJwW~g$A>{mY~{;`"
        "uG@^cFGUONrTDNFeMt;~mZ+GGOzI=4#6jTy-g9`7DBqfO!~9~K=kVNbJe1Nez2>B2Li2jwh?Wyx3oGOk=``<{W>zRk%!UD+tl!95"
        "*$pw4$(prFa`NvVe<o5{zE<kl6qM5ItLrPV?Uc5p7ERNN_3(eg%qIQ+t5I_Jqk*K_j(_>}Rio9GRH|)xg;E$2VBf1=TCM{^$1GBf"
        "K-A}pGGg}y&QX}_tC}}at5Wky={nQwbpi5Kc1CVy#He0Zd;xmtoXo#-fW`F{fsbfFve)8?+l>0VrG4u-&B<~};G_dqR_V7A#Hcpq"
        "%RLPj@uRuzwIm7Fu^72WU-L$3J|nCF&B$_Os@l4&6zD6hq4{hCNdB4tg0vD;=$*Av8p$yT4k~$toeC&X$ExSXQo>uq$6Kaf{|c5L"
        "XgBC+A`DVvvf^1ncIIuTw8t3%1^g0SmZw%>0@(b*+MfTQwbB{g3S-bJ6FJXujh31)9Kp_jQ>edz|H3cO+o4?)if=pnRw`Zgvf42D"
        "#L*8mkAh*?9r_9L=e*!}V-#%SM}Lw0Y3ayb6swn}?=aFq9?d1&@{-kcS@V|LfqBKGlA>ZQNH|FdC&VaWHgX!S6vC8jZB*M^p5-F~"
        "$V`iZf^!pGs1VjnaULHXPUq4u1uRaVkHnZMF;RBeV2KGLlW!UFgOEb5uL9Je1(*$dN)T{DwETzy_)>$424Wu6b9eD5iWV+hj@Q%&"
        "cqy8_6_Oh;c`J-fOdIN8(M!<?xPt!^@R<IE=~`hOmF$`Gj?3CCf7U(EXH#?+y&uNLGKmqkY$&^XM}hbB0L7|+us(x;Y=Gx>Ngrhj"
        "i}o6lqQb0bDC$V&^Cw|R<VhlJpy1Kb_(&Ly@AOEl?n&J)1484z3)Im&{t-)=D_9QU76MN`K9ZjlfrN5L%o0E+sv)S`z!ZC3iLD6a"
        "3>~R}{`UDV-)5n<IAWhDd^tPkz!fte{BE21-)No}8@{W>69+HHgj3rssJg`T&W6<Xr_|;9!1lOQpONN775#uDqK1Zmai)(cj@HdW"
        "2e^%r-}-`fo0#YJNX8@n!THBhgwFRqa-^{imZGMMM9d_Z=#y4`2T(6m3K_%#=<s`XKt7cfc^n)~=~++8PL6#503dXk1mt1?Ailwz"
        "<~(H@y_O<F>4MPk(}L$;O$6|M0f_&Rf<iQ%vW}ZD=wcdu{?G*<r4u1qQE_Sb;e~kYuk`X+s#iJO+>gVQ9}W4zG-Hrb{GH5IZe}T4"
        "0WvS|Kfx!T%vK&k{y{!pthqmN7(gMVTh5Fn_Eq6s61WvH6j@dA4$3m{<{YLH$b9e?*5rp3o<V@l;`|t66|0btHQzZwu@-UJZ#j&<"
        "Jk=;|2_U(<LB3m#DlDcIbFDdh9*?YkJ3tVYZ$Yr*kuVx=>A#X5#&@V*Vi&>;;rQc#cF_9){p;ZQ3Tx^86br{kA$vYm48(d!@@<6{"
        "L2I62LF*p+PzBa1ebSUZQ$EHJ2^TQHWJhchwWC)H)?t@(xYI&2`W<KYEXbCRA?teOalod*3IVpf|0_`7FJ+^dOX20G;(i9cV?u@q"
        "{tN-^afbg%Jmbjs31T^=0e|l?qqah338jQcjwlzS|DW=OwW9WF+!?m*T5No9cZ_{kJTcUbn@qBIuiZnasw)@h{tR6jbl;EDV9RCZ"
        "iR6DE9cvhRgsg1rC_x-#%E*Sd^<4GVD=>)cAAQBY+*gJk$e~yDFfQ>ZN$jtXvY;B;Fcou7z&%^G%is%+AU0Cy&BPCR9*E)LXe_7+"
        "{?Eh0afyNFT!n>0d&4C*bCqkftAFw&WLFQ1#L=sP>xZZvQW~OUHHk7WUU=aE&P8wifNvRp<qqYTdVv|kVR}YF5Httzg7UbFd-7*;"
        "5()(O(`<D3F20W+;9CY$Fir{wWi?!V7}a>6){dK-g9H3L^S1>!{`CQVtd7AbBS28f>==rl$)kRD2FKT7s2VTuoKx~Vo@AZbC|mk@"
        ">nxsow(kIqht<X@%>4KWTpzRnm8anS?Ehk|bcj2roez+6(6Pdw!#JAc&Xcm|fnkt3?*cbWodf&X=^Qi;O26;5)Ghn*r^_AnniK&K"
        "TO|$4vUSVL26Be$P77(X7tH8`d^tPe<8rO2`)z0Zy_stZCHU7r60zaA{hrchnb8@Y;R_w6BhakJCT=PrmOnChVSQ}ukAh#`b_MAv"
        "rQa)HwB<y}7D|&Fi|38zd>*pHR}tJq@*TdljeExuyWx&S9745UIwH}@u1>{`3Uzeo<y{Y&$gMhqtYyU{)WeeFpjbvplHJa@B@1ze"
        "Gj`FcSDY)+uj$k0Yp*p>y3^duzot<`W#8%c4w%76Q%~cOuEi5t{ch7k%|r=tO*g@Chiz}}&>&w~gzYctx&~)%woE^lp{!k2HQn(A"
        "&&-B*ewH#iwQQW7KnPS$11ImJh%sHMv9`^Uwc0N1)L@1nyZGdC8Mbqk?1BdQ$JUZbL`U;LUcy9UP6fn?;!fW8xd%Ssny@###yzZY"
        ">bMlUC`ZCS67Lk$ptlnJ7A%IT6UB*rDBVvU^(GA1iCr}>n!atfXqimYhfnjrvw5?6|Mr_bV8{FavB?I;u9hcnuGng(O=ajGV~rll"
        "F-fxRuqPip&klC4(G0*dpDu?1;@aWC>*hnMby~e}i3|=*>7r}0r%d)?^yACLH1Z~i@%yW+AqIwJmJj(@%TMl({+pfDDT2pqpqm?f"
        ";2tkxP6RnUxtNI=d}1?0>6GyQc4PEMmnOdfP)h>@6aWAK2mk;8AppTi@Igxl008zA001BW0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_H"
        "aA9;~Xk~LQaCyyGTW{Mq7Jd(q|G?qygVfk^+QAG4p3$~*X^R4F6ExWc7I9XkB`W4bA_bDl1(W&jbIze;%d(Sp-^MLskvuwge*9?8"
        "`I)yW_w3*2Z>B%6x5~3d_<Y4g(^M)KUK!2czjCG$UKY&C#yI7Ty=P8~#;uHZ3FrLq;P61zjj^5?w|=y8{pi|dV@)pIX4T8ODOCkB"
        "r8PAgUR7$zy46z{PS|N{<tbRa!yy_W(WNR)E~+b8*CNS{_EzNH^>2P!`}hX5PX^<&&Sg||cu>faIZ?_CgUZl+bR()(&YANzX48M+"
        "_x$jH!3S4ay6sMoj)Q<ISrne|qm5^@x|uSTlqAY!VhQPNy^P}BW#KaP_AZLqH|ZW7JV`iHItVwydN*3WQvOMS#e$aiXr4c^AoP8d"
        "uY?8HT_bXE8)uCzXf|d)v2S0n5~gstR?)>4&o{cPP)@S{;Oc8@jg5G#UE5&MWRY!Ji)z6;1y=gjuHEYEMqNsBE~4j6(R|xf@&ZN^"
        "mL$n#XCHd>ESQ46yYs9zg(|xacQa8GDm)dklJxW=SpHq+*u+}Y(#@7i3wuASl)mm-zN<SHTREkj7aFod?o5I^kQ7{{sRU+J6Fr>}"
        "*b<)N&({e63sxph3Y)J6EoS-uqPOqPS#T?uc=#m@a926@5IkoOxoPg%gHyGtgjN0?YE;O(hvXR~L9*!ii(~jaJpUfQ;+Ri1MnWtD"
        "qcZt0`F{%=YGoQ}DcgNqD=7+T9qcO5T7pZWsv?LA>Qb`l6#AHnS+qDiirF%J|C3KTkI0dDs{)<6C%vkngO$F@KzD{_vhY&x5w#Qe"
        "b8#8Q*wdyeCQQp)?CV_<U>gK*r-H~}{%g7&Ys2HMT(4XLz$<kTjc^H!#PP<d#>!INfnQi&aBMq)(R4a}SRCQMm_^BM-nIiEC27hy"
        "^#09Ew^hYQks!h<Jwye0*TB|*^8j8GGmdJ4PoxO!b0#Z@o-b#7+Za>|4Z>zXXu4GpT(69_%P4S^y?8OdjKeMqu;#v)>CkeaUZg04"
        "($}M1RY@H==_67Z6W-!f7@{vY`lCs5;baLF0NkVGuif}_^v;vLSG3_om6@>_s?2gzRjR;#+@KlXZtBg(hq}Ic#ionmD0(@cz*qc|"
        "L4}9ISFawg^W%G=OI53{M~F)}+C{bjUfeoT-zaNzEwzsbatojk0de6^kGLQO?eWL;oZV2~t|#ySsWUF=6eJ2LfCCj>wAaI~-~etJ"
        "kN;dNOWLM@^dabFzn*-2^X_f-@#O5kXXmd!r2ILAK3I7B{^XaR-=|;-dKz0gte(IA^x@sdpWkPvpH4oUo~QhH6f>CFgo+%f!{W@F"
        "tmivX|1jUfEV=@0PNQ9s4WrmT!Nwpo+6iTXU!es{%wz9HLTG>aH*D+P!>!K7;;yM*axRgx!I|}&vH${*rYHbO$~%%brQ}R`%!DK8"
        "7`kJiYf_?$gcNaNB~YuIEoMat16#W}8^K5+9l%`RyeW%F`HWr-so$r3uvZ7KBgwN87_+*O&^n|l9$DL;&aISX(&eJrIwVGZBEf@Q"
        "nAF#WvQZeg^xU>Ud*z*SJ~P)ed_u)7Py(4P@4a*p1&eTxN~O+$AHcEY%jpk19;nC-5O0m>f5DM&P5|?B{F@s99?^P}+p0`U0cQYC"
        "(@r9(>Fo-v5sgf>xfNFP%k5I-E16$ot)dXd8&on*9#o>Ou=Ja2MPgeW?WVYQ^@v3mV4k$|ORQNiM90(xoG(WZ%amVEc2y|dv_1_S"
        "Zj_Tf()n$XV!ih)0lK6+__<GoXaJ31N3|X;QyzAuys9Ngq~9Ko7I)h-q$>#^hXH&|FYb!qr#o5$HkJbVrI<4g+3RdL4@a&A*d68x"
        "Di1(bLD6}9G5_xP_;OeIaK{pT-S9A)>n=#Ni-|(fu1f4xWtI_t2K_nTiMzjWx%(Pvd-QdCFhrPJi8tQ<-SH+IXh(PVs_uY+<kr~n"
        "wb<~4Z-G-k1`q2D_FQtm?(JLhBSQG;+#h*7xTsEg5m9t75|8S?$3y(=nI262gY6y;%-m|n{7N~1H$j&(N^8M@I9y5pBP1MFytG<W"
        "-BAjl2x7GXsbq_b62O16|M>VK!}Psx8LG+P>Z`A0F^sgM6uSgqO&}6H-6xwvX&H{sx>mp=0{x;w#smHZ!Qqb`S*zjt3XOo*m<VsQ"
        "`+l{6Tk46f?wQDQ*}!!Q;<#mWzrvf1mWYmE#Y>u;g`a*(z&OBN>Rf9eY^N18k{?`p=?0zBS9h!B3%k$*^4cwJ*xcvOpZPGWM8><j"
        "AMi680?7q2AhhVsLm)*j7w*|YN6E97@j|1L3F#(=dlqliGpK~t_z0nWv4Be72;eClzoVds)$2keoynCoZ4*7;=H@#g4lb_zOSj|x"
        "&eKzWO!VLyN~s|i`dq?B?#y7G!?8sIM+|44=9FD4($wL>Ur<W}1QY-O00;m803iT7l-f^T4FCY`DgXc@0001Ua$_%IX=ZO@VQgn*"
        "ZEayMWMOn+FLGsYZ*ysMZ*qAqaCzNYS(Dqi5q@`-{{zFkP`eUGn$uQFTbtT=Tw6));qh)BCKDD!f+J=qk_JF&WM}?+x*Grq9&>E%"
        "K802Gh(M#!=;K2-K;QR&%8G&~o-At-b5F(_zGc#5MdH2BlqU;T$xW%eEmK@%EYChOm6e5;mclF3G|vj|30{@n)#b;~_x;zeUcbt="
        "RVkFWVRDma>*oDGvMibx!JD75URR=wxs<P8rJ~$=ahc~l#%W~8*0E>kz6S}IDutm+m|}6xq~y|4G);!2VhSa&-9N!A%13?cEG=V}"
        "-}3F2g>hLZ!D7{L`oJreiFfyGTjhL;ucgdj?!#r)eZMja8sm!M5Nm#~EaR9JWsw0%WK|oXo~EAUxaLSSoFg;xQO>YBuvQ~-@cNYp"
        "538#D5K&*G%4U|S?!C=a#EV@f%3{k4+p4KiThg>hmU5DXTA^`*io5#pmk-~)i$1*j_4}*ufBfOfTX|nJy`N_HsutW|dV%kXdL$2p"
        "+HjS{vvz4SOFeA{&UOoY11n^OKbdy9MtsNe-^-eKG@Rw0N6^m&vej+Wo`KFqk#CQ#{`qcpi3GEcDAR1QnE&0MI+SnuNxC>WKVS3X"
        "_$^zo<CA%Ee0s7zozLH#oSmLR*PFAm`JyvOug|n}gb{qr3$ViOxF=D;*I6w|9z!$x^|-@yF+Y8ClJI$YHc$Ck!p=_5-ku&`%+J`x"
        "d3?SA87>wqy+~gmiQw`RDf;JH2yl?25AUvKKco8MIe&9DUoYaf^JD=p>x<-MeVm@a+Sxge=WpZVcz&9)j;bHfYhJ(lwh7DteEBmk"
        "Rv$&p!KJ9piKdTC-YvB#Ss@iGVjgA5(vwOUza*F%G=TPYnFPRadPJGKgy<s9>zYXp5w_iC%4yA{tldG1iCeC8G&#+A7V|=O2)R^|"
        "j73&ycHV!zA4-7675y<$c2JwLaKrCWQo%pfnczqxA0^6Jaa*%nEe7=XdIDeTJmD#=8T1ESGIa<V4NN`2aacdFPO?~CQxp7m)177F"
        "&A!2AV-T=uja|ql1lh_RuLOq}3jB>ym0TVj0lIAJH9+3>$e$P*S*Ju3LMJo=zpM&;?9a<R7s146KJ|(cw}mjR^L&d64}^csW}oM?"
        "w^2B|`TEFx0WTwYCLZiVei-`TFS=#f`Z##F<le8?`c8;a1b$t}x<a(%Nu-4(kn@cYMlG;N`wO^#c`%;Fc>z6=D=YJU1Dk=-3rug7"
        "+6n-RYI<?F(={$18I1Tu@gdH+70k{bvLp%g!rKmO7&HGhz29`z`+HRUtkhWngjo``(p$L&2LLvP>AI!a;}isZg!@}0%6;HO_Y~&Y"
        "YyTua9LI>%<}J7|P;!5Rdd3;<igI7LTdwA2rAO2Pb3+7yGzMU%ge#!!{#4g`?2QGI2%!}HMGLrE8E|dU2xE!uLjV|K_*&#XuE`GO"
        "IY{ig4odjsjzg3bHRVbpLm{nN=X@Ihk}3id%ZC8X<s-|yMkX5S##m9i8Ln~D9q+(tBP-!fnsLz)I2mE?V7=eP1D59j=-u=OJHn|6"
        "rKdZCB@@a;cW8csxT)6CR4;OBgq=F_OeRnI#TLeA#cecj(~Rdy<Jy>bLe}L7XD31lp9D3CFg8p7I=j62axp!fKloP2r{%I?HlpjH"
        "Fj^ChvYZ~Hov|*Xv33DZFLJ=S(Xh+INiOvSU2NzEwv+fc+}cJbBpp@R*mRm8NIMk$jd0@h>NQ!wbUH{?*3F!DttSVXl?WOSH_3rd"
        "TT{O^D%jR^S^V)`&JDejPyjhNl7^j?Nv?F@;4b~x;5BS<PO9<SnYcMfr0!qNp{os(q?k3E6lKy=_Qc&BSQ6Q+uH2K)Y-|pM6&!5`"
        "widj`4dQ0mcp8El649tO3h^z}nFt9VVl4_4MJx0kvH?w2R=fxx%`OuFft6pYboK_MhoX1`{+{#Rc8K6ZRWRVsD1i|HX{;-aP~dYw"
        "fT1!O)WZ@P;25(Ao<Ft>cNG{OpaF*k4I!OkEje7*Fh6ht8re-ja;ctA0*uLU>d3f`jF=GY;3hG8iB0|f)Y>X0KGj^g%QKa3bpXEQ"
        "-cq1xs!!-Yu1(go#eq?(PvK9R5Nmhs7-0tf6RmXt9njxMW}o?Lar~ygR7YeY456y74ycr%v5kCr+G+HZf-Dip37dj%yk8eX?=xl3"
        "6t2L*Eo0yJe*`DnX2{X2U{ftccAFI}_kQ{C57^zd+gdT$?Y(tguH|3AlpgL{x{k_oN@Wojq&=k}e}RCDqO8bN6a|v!DW*52a>H%v"
        "eTDS`#A{S=wJ*ip(p#5h4sLeIVBxO1kVA+JY!u>eECe9E*zsV}qwH{{1lD=CB7Td>OBQcA07v3T4e7gq1|ubuYPwb;fLI??Clv=7"
        ">Kx^2Pn$wf7eUw7*FIJ$P{Gb-@MD%_!l$iFdySdoX4)MDD7)&Nt8{h?d}3D7K}K1ts#a9~_9#@g<<JWrICpxw;i(r9>YWn;xeql1"
        "&l}>5-yA!k$SDIig!<a|zP8MzO1KbV!jXSqU}8#XXwEP<zFwZp=Qmx>6Iao+t+(7@QLek$`Gc-AOvzi9y>x(_Z_Dl}T)?_nfhIhg"
        "&*!>XTe)eDpacJzO{H5m*r;;x#AF1fs%<j_lqIAzp(Qq<w{pDv$P3INXDB!9`0PBe96|zGaQR8N;rB^)%MmU>%cc$<pswcLhDt~o"
        "at^M)<B=+ZX42=8ShJ#xE8-G7k3T*izXu$XgB{5b4Wo2q=h6Zow*FA6%d##KBga*9cC0mL@@jx46Y(hrIE5D2a-h(Yh+awP%4(g1"
        "79MWzARZwS@mxyk5b{X=sJycWN*%#o6!Q2qVISv^%INslLc{NL?0DmgHS|NMhayPmA(K8n;PVoUz~_0Kz{g?*1mg73Ks0uI4l`O4"
        "K8YLnwAev<`oqwpC%6)ao{UYsIU*SO2ki!0{WRHb25usaC3jq8>A}|s6FSushU34#=-&^0zr)9_W30=ha*wK?-^aXCj#7Vbh>e6r"
        "ci9w9xMZCfj4XKaf1uU^2<V7zM8aIol;B!Ek$-OM>QGiW5U%}y&0T?LR{>Vnjzg*v(j{n{chgCn_kii75n$Y!L^WbSyvoUnx@SYk"
        "X_!v2u_}Z)X?QCu;>x0@iY+K^>fL2UGWAM4n8LIbhFV@`^biMdi1(+JxArkX^f4VoUiRed%i3Se=jW%VFem-II6LznUV;@ZDnPk%"
        "CnMN;0C0hK4a$vxebgkseP=-Zo27RzoC0j^p~$2+SR3l=74H54YGJr$P4Z2fD&VSN%76?%@xEC($;EG4D(=&i$FJ)IZwN5jc`{rf"
        "-QiJoGUBc=Q!|_!{gsSDCT*glzur;LqPxYFou{v|UZScY>6aoyqQ4R87&PJYXYV_G{djbB`O&NKES#o-viSfzIOQe4kjHdv2wOT&"
        "q{lGcs+=pl69&ygbrs*iMiss^BEV)CqUew~+lwQKEEuv1O7|q_Ue*TI$)?0|GE7y6DPo)dwa165t_(9dagHl0pn{CVJuDQUSdkb^"
        "9nETWmvyG)hy;d2m3O=V*S1GsUzxw)E%6=KC+=AEaJs~?iN}j>cML81zqx3RXtpMuKApck$BSRgqG|DT)-olghSg15oYEyJ=5qX="
        "YIHmr+++!4NgkdQ)%vReS@GBdb$NS{DR5Yvss!;tg$$fVn~~~M;|<ync;;kr%3fTW6R3w59hvrza*?BR(%en{tzC2hfSCDMaq#5a"
        "4pzWuNPf5O5xU*}?Oe|OY0t)BFJwYVJedjxJNe6|*2II!n~xA4FMOIi!b!*D@Wy|5nA#H4UP+f5V5^RoK8)BD&v~>t$zNKr+QCn*"
        "lWxPeOx(E!{#1iAAxJb9UUQI&({~mKmE&JvG-hw^I4CIWhZ*R-tH(dmaI0Ax!g`_RxV?>J3OOBWt6}st=b;*e-<Nsfu{7oQZ3_9a"
        "4kT2UzMLjRh75v3K4yh(Wj4r*Upk=e9{D&V=*jF}dv{LpjS723?_i9rew&@!ErK;aQSRHXM+ieSY)0x(;a4QSqhmeyuKOj0duOyX"
        "s&1QHN^_}ai1gdAzxjlqOBjt@F8Bl=%hr+IzUIK&^xOBo?Dp1f>AHzSV@wJC))QTkt><bB0FmS4F5#^x>k4^Tr|(WdcSy@o6zZ%O"
        "ZSfcvd*5N#t6xW!5;b0lQ`+AyVe8=N0s7$2SYLZCW|L`M&_D=aI?uFT@if4=I8yGfeb-(68*s&e6ZZyC2t|Fx-#h^*G-C3^ye^YA"
        "zJ^C><$mgX_F;C_Zt(nZ(LmSD*0&Mcoj&*G@p-R#xsw9O8A~jOcIbSx<lKr`(zeN^9DZzL2Ml|<%WrSJS~aw7U1xJGr0vpXhFpI<"
        "XufXHIUx}0tZe#p_J_xwRLHdl+MO>F+wUKq-OVU}LC+hw?#ogL<Tgkk+RV9?Dd>5+>yqw~m=;ERq1#UDb>F+GaEGr2&mLcM`{{<Z"
        "bHn&QP)h>@6aWAK2mk;8Aprjuz0CIr005>O001EX0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_Hb9ZHPZE199E^v9xSzB-2HWGd};QzqY"
        "zDVnH<hFZ3;o@v}7fHGW7TqFo+lRzGpe4$OjYO&>75g;Ef8UuQB~h0gCtVb%9}<g~;czbBd_yNm^5m1huv!<qW`$PH@WQb-yp|<*"
        "QmZeR(vETV;)kbO+emj{rM_3S=A|%98|H*{?7)o@*5>DD=V#->mb>D>q-D3=-BzlyE2O&1+4TX2N>R&OVYn0Zkr|L-WZ^^!hnWV>"
        "{sn{|xGHO5&(FTQezyIbHM*3QEJ!KKbj8Gjv<@V)O2V=YgXQdvFtA9#k$O;|ltvVO*FCEBm2{wJk|e0DY+7v`tf?wcX!t&Gdx*b("
        "vRVxvMhw5&?zT0$fV&SvPBblQbbeMD-7v5sFKTWrDAga<4}~S5CLY#Le+!?e8qM9x3O#ctnuh1Sdj?6*Ma!jm`oNpE790H3Ryu8t"
        "&*b#`h2pI}Xoqso&q`4-8%Fkc)wKV*TG^Aa?cW(V=<588!B5MN=zX?hFViYV-8ZZv)r5kz6~sivq+%8n6y+c}4_<mLj=P#Sw<TxQ"
        "74rhJ8!uW`tQIPL+me)a!}cgB9QZC{f7sD{7|4P`_+z-Vf_=~HPCPY6n>2aSwKYbI7;Z{-wt15IZRiEK1bQ|vyQa0h+zk`Tc1G;E"
        "Eu`E%<Fy4_S}-+)t2TC*CYxl#lB*=ka-j-ciZtn5wf(#oZG`KL>Vqh^2maY#|CZu`S$+@?rM!cHPc!d~e+=P}g2#UmYIkiq0h2U_"
        "dG`Xd>A5JhDZ`bevW}}l>}7d{t{HD+h1(nPMp}sXXbLj$?g$grVr+;5uVplUDkg#&)^=hlaZfoH#y;SB-}L!ju3l95(7u{R`fqaG"
        "N<N74#Qbo%<&+`i-Zwj9DGfr*C4QyRohkwN+PzlwaSu4V*A+mV<n<n5)h1)Q?(d}A^R}&zsmC*L#uY)uhK(lQus>};8$dw|HoMij"
        "hSNd-7Y9+io(}dIRA@jKMJb)YIT+mA0{;b1h@5v$7pAUu74E@d(t}oQ?++4GZFwPfO5Z~nNXA_56&ces9F&6qp~@8Yhc$iXz%H-n"
        "sXeq}%r)E94pbkZI5(Y27XsJ?+-{rPY#@8u&h5x*^ZW%f-Iu$;!tlvkhghIHhnp~VH_XXKz`*X$pZHU<=*p-~?g0QLpTwwdsXqCd"
        "<qI3DIKU(%bpq-JVb(#ZHF+swO~LpObv4)>P-qO3p?vP`y`U-#S%CSJ`v0zGctQbV0Vy5D4?+DSDG7!1xpx|-M5wW#Iq!>a8OQ`B"
        "+<+orwS!Q8%`s#Q@dcFmL_>8%o*`8Ql^+t<GAe&P(6O(dK<y9@fJ1@_$fypXn8*i=!*Ak26diy)+9>xzzXKF+TdG1(Min>5OJ0_j"
        "z&wN#mqsYw2(LaIj&=tRfsP;*m<O}q2jzgA3?|Ko`w#9$mAzr}IU1o-(e!{wvvGpZ9O+M+K2p&90jxiO%mF1%x?pJ(j&V*R88-q("
        "e(oTY%RGrAl+GQ>@+}D6V~T(rsIPN%PpZ^c0~QNuIv|9y1mmHj!Jv_PP|Y2(zbV^{)tl0inX=hJ!#P}^()M4Ic%<sg8fhECfk=g^"
        "*F36&hq!ST`D2<OS!CNSASFvq`|RmMD~Q%|WSy*2K0i>vaN=+UB1iyaZ#^lx8d=pys^;FKJ@0~`p5>yAa!TsdB<Yt;mVUYWF;Y59"
        "Hj4`eWeN!WJ3*Bo;5Z%~B%=awpc5wjMl0e({>0b_2SJ?y=|H9h9ip@OW%}~RSN7&2W4FVvuT(M!%SKVb7eONE;eUJi=?%s~`1tH5"
        "I(*#=|Ap!bw*R@)=!(8=*rM%bT?Ho12pSZn^>OeN{-9^rZq#I)nUe3+6XPVA-iNCr%osO`Cpf?pUKm0fV^yd9S&l;!^RC`-I!z$;"
        "Z&^YLvPL*gM3=ZP-49I~0i#b*pO8R-u-Gx!(t?-sWGZ-i6|#iK?0F2+U0@bwLO?z0BVP(>*Aze?(8f-0NM7<Cki!BIT0?=zXNInl"
        "S8BW6LcG02q3>R)i;D{gx&A8|RQF-`sDl5TmySzcQ-%r>z!p*+XbAMWlre)PmtinHu$D^1L6Ii9E-^X<!&<Mz{rC||mnUfLk$(B="
        ">+4YDh!4b~F|<&8bX$)OIB3&o^`n92Q^|=3_tA+H0->jI6y6iaT}H|Zo;Ixu79WIDPv1A)`L)2#7|igb=WuBYl}A48lDL)Ll8h}R"
        "9=Eia_C#k76cCb~@^M%c-RY0)2Ltq%A$<=-i1O?UrjQ}Adz!VnGr(YowE}C5*$VL&029-bi{4+S(+9%jP9WzUwY;P)Ys(NrAeZvy"
        "HLOnkhYduQ5^Jwt_dVIlZo0!BU7k*}{3RZEgUvE<E=bPZz)T@nF|>9Q@LOwOmSt$@UV?&Tw7GV_MpU%jH+p@wEoCvkS+v&k)TkVO"
        "n1d)L)46xDTtn=qCk=-22y9UB=4y%Rk>=j|*!Iw3=s0+p&O8_rYt3!Y_mrXf02XFFDP0^eTeqUqGD-gv*OCzz!zH%>hYbJ~;)H+L"
        "3;98(P9S4+K#v;LluCFok5PSnHv}*4UMe4+{qjJ}_V~TB41ZRw7{TakAy7r1S=nY?$kM(aEdjbB+`p@m{{G2(mO334-kw-EGuR^Z"
        "eJ<gfMUwxdrAqzq1O&hLJst9hJE7=Nx9l^AN*X=VVmpURpk{ZA?(n1FZzZvQYm^#I@sXG`Zp?lM0D=uxj2FnG7csfAkK$pG=>AO("
        "lBgVdCTQ0UP#<hfed9yYM>qePitvBbfZwal*JAX8%?^F+Wymg{oWncfvYqwj->tmhrg5fEn@SL_BdAEA(~)mPxg}vSh`6qWPG`08"
        "nfyIqG8Fx7APT)4^0xspn)SdpeM*1dn^bg*$qR8huLJ(EWqdZDL1BV_wYmi-A1;T#U(Ml?pn9_^czEPmKNcTk7E=+VL=1ehaLfE*"
        "X}Mx}ta;zm?3E;1Hhpd(f5a-x{(`Z16`8%BPX4-aO3Zlo4-o;O@IO#X0|XQR000O8001EX&twv_wFdwI)fE5$B>(^bb8=%ZVrgb?"
        "V_|G(Wo>O?FJxhKVJ~!VYh`X}dSP^FZ*DGdd8JrObKJNQzN<?A0SjL^8;X*=o6TnIan(NTln-&XY*j9@iiIIT4J;C10ML@+=)b2M"
        "4-({zq)J5>i9~m!A76jn6s7d{vme&vE7zhCye6%xSWB$B;ZppYYr>4-*5pJrji>}1y}uxqcYVt&&q>d0wL4L=y(gCGYmh7`rN>9d"
        "N22SMwq(c5t`*zy`HNBV_^8pUBmNB>7MdG1=!%>0jx?g>CWyfmXdxc`0gq_komtJKX_W4`9(8{ith$KqkCI!_U|`q0>sV1K*@)|?"
        "@Y{}8H@y<le&kI~-rn_)i;heCspr)r>Tq#o$16Ybo(cW-jzL-=De?At_8wh1KB{>`&|zjvE-O`YU7|UDXEW~Q)^l?!a`J{4tJlXz"
        "1b#G^paXT<5c*kumbCZ+o@f3Le;WGY#S8LIprLP=#{v`h@LBo?cbT*gG|^DnC7cLJ41~k$=t^NE>&;Ht!WASZH~fCnvTj>5r?DpE"
        "1uJiF6}_~Bmcf_z#ta<pk46PPKcMs9Vl)qQ%1*0TT}C*TBM`G{EYo^fvQen47>KPMNLC?ZfXCnPEzwF@g7GsU4bklO^jfLwmY1a~"
        ">&cHNMNz!+op4XK_ZIND<&Yijd=m1yKp)ZONB&h9p(GI?q6S%m3&c6d>k~+$YZ`}r%$kIIg>1OSx;Z%2Li5TheNQCoc;StH;PzJO"
        "8`5y-DY%U=u&a5eEKm9oF+&*mBsq9WxMbUw*TopC|D$74G~8H#33gn80hkmeSkH{kVC_4dt{kGUrWu_Q8x0m{4rDZkDtXu<jKSRv"
        "9c***Go2%WLpN2{;We7LoGMv#5KUW(`pQd-W>mPOIwYB1k?%GnQHlraDIU#);pCsJ9r#<VmCoo$0YoRo!cz1=H_!qIcMGs!LoLw3"
        "F~S(ZRHV%kVdRuh01$=W2Z8|yfgUTK(NmfeT9(lwb{ZBu2iw<J9dKca*VTkx72r*0vQ-it76Y7^96EJoCIQHsiwxwTy-Kt+GcctH"
        "iQ$8??+~;;Qw{w)bR^;pD83bvFUcQx#8F6$B3u}ly)ZlW^3OkI?rw()*E!j0RqXhkk270g1^wR#y-Et#B$dM}NGD8(Q)cPL<mG10"
        "`31IE2~`G?hH&Zx_G1QfKj&PgOo{VrbD;+)jMam&rSG~Ddu{WMwT2&Jx6y%oEaJps@eoWQnU{h5ve*@es6O|H25hYtV<=z5oDy9)"
        "`#z7Ye>aX6CvVn(Gh~7v&;I&TB2@DygxuIUJjf<qjpyXkyNei;c|}^$2^fx=e}SYB1vJ&P>K5!93-|z-5d&$L7MBJ+$couHx6byD"
        "_rDuYcaFR}fPetPBew!t0^3)@i?4uLA@9josOffS8FUC-a{lt?K#Lb{H92WC=j578u0aR*0IKXp!TYmwPtuIkL)W2K@E$wD?=)lw"
        "Giu4Oq2<oV>jN1luQ%lUhx4<)yd>E2iQGKtlic%M7Sq}RKw&Pq3;=(e7lpSl@>cLHj$0o?={B|K&7rdbmJHF)Xk*_KEOe+%%;CQV"
        "Xey`?7?!XX+PDe&-Ed-4A9x=BD3cNO*vQ9DC`QEv_=DZBJ%N>{*Qz*=!l|eYsAP+`iBBIOnlYcwksy@hC%nlI=|36MsCYQ0eMjwp"
        "<PqKl^(;h~6IRzH+J*Hz0L!u2YEIaR#mh4{Q=`4DGF}hPp5tfwOPH5iX3|kni#kr%9=1C2>pwmf>BQ)PlzUeu`qA^J`)Xihkp9D?"
        "77ftmj^Xxy2Y>=}!-9>wA{pFx!fp>9$Ep{5Yh~z84xGLIzb@zg?!xV<;kszO<Hj%;W?=C08`Y}ibWP}BxW;)OR&03XuYCy14P*w4"
        ";~e*hE?D2U_hkjQmhp=!nN{(U2_$|L`EpM*R{C4Q{PO!?iPUg|OWFSPLK1B`?3DkjBjitMw6GcoV;RIb3;7VUJY40IwM8(~!QrTu"
        "QU~M9X8{KOi*W+4F>MXfW#&upG{q&~x1?4)*%>v?ncZxJc!ML*{o|}>V~#`5dWnNA>q3b=`+{7+O%(%D74CyPx&*a?_Hum7Z{51a"
        "8VenPj#aJlodO6i$lpdcqe){`51Q5qPdm&jh|@*s<{nsfZxg$y*P~(gtpnpV48vw<+oysZOQ6GSc|L9^UbK3-Ca@)BQ%+Zow5I1L"
        "%lPXbZ$liFR<OqUp7zS8VM2}Jj60=Atq*DRW8no^U(JA)D8YE0o-R+$ox?FH&?GPqYzP~Ep-@I&V<G<YhCYi*fMBmAymuQw0~-o#"
        "$Dx=LP*M!|1f+|>6h;^U(YZ%Pz|-yF95Dm3$BggcBFtGba8~5ap*0Eeg@Q5DQEIW7KEE1mhZ^h2?<&q~UcZ96;QwY?Q3*TlnEne8"
        "wrckeqk8l<tn-JL-Gm3nK_ij(Dh?jch-nNqPX|xTcS?aHMDmnnEvuPjn{V?0OrzoOdA{CTHu8l+%-*Q5+_;ts3r+=%CF?pdt5j<q"
        "J@iTD^<x5IgqFR6s@%H?(#G#E|6eEgaKw4Bqx=4)aG>iwa6OiRnZgLNrF5v#OgL%xy-0oaX5v#t_r&*64?6oV52@{ivHvnrdzhx+"
        "e*sWS0|XQR000O8001EXr3N>f=>Px#+W`OoCjbBdb8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XUtei%X>?y-E^v8`kIPB}F%*XP"
        "1M?1ttQ4d#ATF#PrO=Ktbs>m^OnRmUbK^`ZeS8x~3)T^uRmgwI_X{Cf>WRr~f<Xr+V2Xj0@muHv!`$>iD|=7QsY~dx@69QG2ilX~"
        "B@;ZD0enbcT(soKC)&(}5Oq~oJqHKfSg<VxGx|_=r%LSWDs|XUN#+mXtgav*e161svBKs_t{!f2xxN_RTfD;-n;Vq#>oqPH4gM&R"
        "jZeACMa5<aUMAI9`X%N~%J4}Z+DBqjh8#TSWsH={S}EZQUdEL8Ln^-e#02X>o2EUI|KXi>9TOMVYh8T;P)h>@6aWAK2mk;8Apl&&"
        ";2mKH008tC001Ze0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZBnaVQgPxZ)9aIaCy~O`)}hm68=3v{)d5zB6i`&X5kh+=Umhe"
        "?bd)_u)W&@4uwEVv`s`3U6OL*=JLPakdpOucC(8f+Dj0`7KcO5H#6Ty%Cf8_J#e8wMu&sXs0RbK<u38x(;y%qT`s5u;`4APnMXm="
        "0}(_~{{)gp!o>-KZmFQ5OJ(c&>iTNEXA%PLMLq?O<K&PlppVpz6op@+fQ0su_zjp27K;Yx&kH}24d4Pj9k^tQi<ACHm;goOy>`2Z"
        "Ha?SkT<yFgLBZvrqK^uA=%4P)H^F@$ZCoDUE}7(^L`T0z42=a9Q5dptcYU=XlKLz}FAq%ZLG3Xt65wtLE=N@Z_C(+)?g<|wDB-RF"
        "B=kV>dm6SC=RRyAZ$}kw%(5&U^j7e|%y)evC3<FoK{b7pn5feM-JM702gP_u{M5rH#(t-<Z(%@NmfG6?TG-3+)Sj)KS$}G;^qlLf"
        "pK}YCe=tabTN#|X*H?f)mQHh$r3Aq{nDLN8Kna2rh2%(>Pd1n{%!AI@kqWle^<Xwel5Dq<c(qf^G4-7gk?Vp~B7V5n38t7-=YYGp"
        "?WQsQd+T$eeuJJ_H}SAJ9A0DRvm<qJwGHln?;L;lw6%PoPHgEjMPF-sj`!2{uHaJYt>_Rx<Jth`L?Um8IAvD)_q{!x_B+mWKC&my"
        "u)iGPsI^z>AiM20VXYnKJD;{ZdTg7jXnWk1ZJoh(jK52S=X5YBt6%MbJsbYwOzi&1UJm9ZXAch#EezL0i?tQBan5i#U#*6?w*9d`"
        "ahCRCo}aP9?ua(VmIUMv9yX=RbP2UxSI2KE;TxD^`(@>l>rWQCe&kW_4(lr&KQ|wQQLv#Rw$ymEXYKy=)zW@vFEI?;8T427WIVH1"
        "UGSKzG<1zpb!#Y}k_tC`)$BCww}a+*ZFNhgwb<;EX`@nKy-ug~rjg5Fuac(NUA<9Wz4Q9jJ1?#N{oB{C{v4OlH!vh24;hjNGTn|L"
        "SV+*wV~DsRxtL5Mq;4$4n9Y#(hLbUJSdLoE5-WuDGTnG|3(l5{fGEc#x`t&CReP)s&7U+UB=jSwibMz&X4nF@1v3dyCJY5pUsBjw"
        "PqC}k?e^KqRyzn5Dkmd!M`@Bu@I-_hL^tx4%oj<#hsH&g()FEJI8y}pXC>y}CI6X^qNFrjN1V9PoZo|Y1h<L?COp0Oj9wRdpE)Ts"
        "5T}}Yrd+a-6wLyv1al>BSsr|H0)j}i`>ivvuBa-k-&+fW3t>kY)q<4u4vz0Bs^p%8n#pA(j2;_AT8T5<1L{Zm9Pa5UZn~7J8-|Eu"
        "jVl*pD~&Mz@W4E*rXKv`$N0O+PtEFC@LfDsPbiygsD{iMPbME$>t%mE{%8Yo&C*^$Z8l%S)b6i7EbUR<ltez>xFW38hs9ztwnxb<"
        "E5G2@0(dfOJa^$K%ki^SFYF5-)4hY>io4v$(+Ohku`LeVA$C}Yrl@kK6{#QKc|&IqiHi%SPOXybO33G(J>5+8er~$|eEZ`w{0E+H"
        "(o+xAwA%fATi1hc;z4~5%0IvTUO%4<2XOoWYTwpN%Qo}N#v{5ibnp#Bu1LFZfkj)Wg@LTqyBQV59<%Spm-S?N<-GS|w`>2clEl}~"
        "p;#497ba(tUq6R>$*f$4_2-=TH1JE1TnY@glB0pkq2;)wg-5&EO&aD+{ZfdP@G1^;sAg^)<`>N^$B%x_2)wci<MVq5Cg^F1o(9^J"
        "E~%8GuyHEFV*01Zz41x>|A>Ok#i4R>A^EF|VUFaY;8P=s9wWPuOU2xbhg0g*oY^fMEMtbRoF5If;?mqdYR<*=<lTj{qgKD%>PYM^"
        ">H<;jzF{bcq=7_u*pi0o>EEA>p@xUZ{&)#(7|y4QiM<}rXD}M~-_Pc&^>|qSo3VAFCo8;-N9v}@E_M+n9LYGx+u7hpDq4lXq0Y2!"
        "I>Y())X?~)?)~e@yjas;#^u%GPv>7UFVz2EK<LF#zn0ajL4U2M0v=yZ@LB!k8U4~Nu+(N-t5}~GNf2FCFNZIy1AWw;7vSNxpbkmA"
        "&gSHpeB~vgEEd20DfF*zHCy(m-&-Ky%_B+{2~}G?@Me1RJTt3`yi-c{ruQX8*mK&4n_j2JBdNQva!G2zM}1+5rxa_e@zTqM_>5f?"
        "f?8Pynnnfy0d)~ivZM8c_xpw0Lwx_wE)jTxxGP*hP=_=<%4mZLRgp57i1R%`=Sr3ZB?|9D{t!Be>mF*Rs+E(7*qi4SM3O2)0V(>)"
        "v6LVT(K5i<LU4h_m;lcsWpvuSjbo2ZGd0c;)}k~zTr2vmr-l7L7Pz<xZ{KuEeU!U{yqkYmg68KYb|0dl6>z+6j_ml0*2ER`XCqku"
        "I>!M>sk|bAQ=NUxRVIM(iNd_b^b!da@k`sv`&CnJ0TK7IB!FKnbVW-0^HeWG(-dXO<`;V{S3&2ybFV@!hE+314S0I4e^wtqD<jEz"
        "ua}UE<o|+~PfN(>J-s9$FUUxY`F{aWO9KQH000080000X0Q>}I39bPE0Qv#|044wc0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvH"
        "FKA(MZe??GE^v8mQcG(aF%Z5R{2zurHQ;X2>mCYQVv|5fLY%foGnQs;VLi-@ls<laN4s^L(&!)<N%Or%P17u|K6QX57xch%M8~jW"
        "v@Y(z#Y{Y)p%&IeV3U!{<+uLsYyW)vaNh>oHcd01&1VCr07uMw@3!E=F>!`REnn}pZb%0GjzYlJq?lP5<@9v<g1%sODLzp>c&|we"
        "4ny+x)UzVYh3P)!Cqnyc$(*>-&1c4=2&}&^zfd&$8{|kL-FybREDb;t8-19L>Jsw90**}9nJPmz)l4^fSb)WhLV3gV;zVydkjx9<"
        "5;x9{#e{SLA3y)+r8~1|SgTS+!*wzT;))Q7J6QF%cbn&j&11h_FQD$H3NWhUI4~+pF=I1TtI}4hC*<o%;iUpXz~}}NO*a(DI#QF^"
        "R)=IVS<r_E8DY0Io`8JNRALG)VouziL3?dthu-~Gsfjh7`X4Nq6pqMFQdF?5CvXBWWk8T(jAlI)e+pS+X#WY|TrfJJcgOf<Fk_eg"
        "&}IS6-_Z8cK!JXeDV1O}!z9IL)xn`E#_2-Ss>*&iooxGt`(i%(15ir?1QY-O00;m803iV8>g^7s3jhFcB>(^{0001Ua$_%IX=ZO@"
        "VQgn*ZEayMWp-g~bzyXAZ*DJcVRUJBWnXZ4bZBpGE^v9Z8t-%4w)K1N@;{*LhmxzzI9t2hsZ~1HmTTQ&$!DqEa&C?XlaR$TMX~^B"
        "TlMn4?>zt{K?+ooZc{E22|T>_@P6aLp6AV}OdlAz+R06klUtU@n=R${M8pM8^ScpYPi2<IsU&fcFnD8eCDS4wk#+jS5|XgI*rqv^"
        "1sCDV7cXC&5K-Nhyoi|)uO3+bKvF^SLJ~>@tmUcP5kd3hws;E3iU}zQuX0l4*^WGJSWY<mt%6~e06Nd}5NEnA3l12BY@Q3zJaX1N"
        "wxpXO$+qP>&03hMN|To_*1Xt~gi0D`R0t+WLnBFJIcld2Xi4QJOK)qiD|kiD)ZA^lE@GP9vF(<Iagj?-W7%N5W@RB#kk4+J(^6~-"
        "=~{;L)LQl=>y<Ihi#!F8NSKw;hKkMLa^NxESX(K}Y!|7)NEQtaRH@=Hs`7N5vLs5<^?C$KS+j@(v$#ZU?&XWWH)8OC=I<<@ta!zO"
        "moL;9(Z;i);{KY6DwE@vF9>|nTu7S7EJ~9x5t8ds7?YK-1~g~LB``52w?&cZiFK97Xk)PP63FN&@^3OPa;70TqoT+aE;1o#B$`4}"
        "RMIiZIJZafU&7YP##vM<)7NtH-5VGgOh$rZhfWPh$r;*DYM3*l66`q=F;7cr5C5rNq&d*OMQ<}^1A+gs7`&mi0kNTf{`Rjn49&9Q"
        "F+vB88V5B}n~-yw305<bZW)Lu0#7XR1SGr83Mv8UFUNgyhN%a}MZFuN0szM!SP+n-H-kp7?o%V*Psp*ERZ66(VB{B?Rcy+6!F{h@"
        "u&qE|x1bsnG9dSY2IqiJ`1M;rzL*Tl>=1GG0523p*F_y|Q?W(ksJW1QXCcyIa5Cg={$aQCTZ0LLS_3gFCGlI{Mr5%x4{d-8S6W!p"
        "r!oZ`0Yoms5XjynlNHbRG~`?MJtE%K$?EhXIzO4cot}9kV=wg&sNGj+gPZZU$H(tmO{kdeGQ5w|Zy|c#2HY%HC#%_Mw3xqr2h3o&"
        "8EJDPb*^eCkt7x9E=lOM7`qHwo)^Ma($Wt;TRyY-`SnROy}n*t?{QSrnG*>SL#Q^CFUe0umgqP)BAYaUWT^lJfdoimP#eo-3W6p%"
        "14kq92H$eS$QDGJfn>@`h8o#pDmP6s3s($dJmtGHP=1Vo2rUV0apIZ0aD?`J<SDzz>6Y2sE(-K{%waxG1tDjm$R3y<Saw}-f~0^6"
        "ZQ~)niE&qVm?{93uTZ|~<q%)7icf))grz403R3}VGo=i0A%!#O(fg?DBBF?in3he{uz3Y;I7m6%-X|&d^^2HjowBDCjHS4*bKKtA"
        "M<`xcbh`r;>aD4K4cMS+0)6>jCD%uP^8#zFKn3Rv<FNPR?0j*0@-~|O0$=B|`D}SHJ&Qc@wJJ$M{QAZZ!VP<})+>vG)keU+7a9FV"
        "X43_CqJSYUEO)jHz9j!(tW@T;hKemCSm=ShRXL>7c!Qoq=o&4M=7doW1sRA$q5)`g#7%}aK8R!z?K#}iH1`pBqQXXiHQytLkO-ks"
        "6l6`)42sEDV6<5MRUk$IktNMCHD%%K##TMTFylb&hq@EYw_MuVoQaytT`uX9G7DEgJmmEIhW5!}V3va}(u6cBA$77x)wChBOTv1s"
        "1sS0`3fS(mR}Xm~S}+uI10tC0`)iIRI;abU^+>aOABER<A@I0V)LZj$$w)1jsjAWCa9!@gs)VYLIpn3VLx>8gKu7`{saN%~HgCuu"
        "5)Jh}kedc(0$$SWdmsOP=s2*EbEuI66=xMSu;41|QE+UvmGJ+0n@GsL#dc5^**_b0FVV+ve{(>);xrdPYs+|FV_4ZffM8^F3hCJw"
        "EQ8`6S#hZrXTv*|GoHqgTBQUTIGjO5$TJ4}0g-fd5n>gu6>=gW^WIj{g%&@Jl5j~#B^K(#BC1!o=!NKj#ti0|**=A!S`&i%!tT1J"
        "?$8fv+aKPlBHF{RF}IfG67IT|uR}-pQP}R;?y=wrI{(eRvb6ig%$hksbstiGhjp-rA=!HmG3dsPRtW>xHmiMDH_?kmM2TgBv!{jH"
        "t@L$p9}wz$_pk}Kh$k+}U6euc!c0A2MKwptc85XmgTkHELif{T(&k1u2FLf2N;I&OSUxGZw*^i7AQ&A+3o;Os8x-pMk3iNkpWGaR"
        "JVZ|gE%&rT%0`s9$PN>-i3N#ClvS=%B|by#h9?-X*t<sd!tVr?Ak?0(vW2bQ4dpo;HTHeptNSfG@oV2l*B*?9x}eJi<I+DKk!MVT"
        "s-XLVe+c&0co?)?+JmaLxZ`vijQ7`Qkznoy1`S-O2i(Bm8U`N96LpdW7t$l4D$B2{93#&$QMpvsbB|Oks~8Vm>nihySW)b#(sa2K"
        "5bLMlM*eHZUM*D`uJ>V}z$r?sd+xI07QD{i$FuhEAmqozwuD;Lf40m3WIQ5YeXma(M@>`jNNw`XS1^Z5zY0b~({Uj$c?ur?To<r="
        "YO{-6eeBWYF*dM#V>nP#V`U7cH>qo5yL`M%^K=V!D9ayGUgTRK^&QBaT%5o9DU}!1?W>ckS?8F9$u-|FB$SY=cdLuV98RwnRspF_"
        "q{<PV@nC!5;gRy)ay7lYUMyDl_Rrb;Y=$MAce=P-UQJIueFnam{!7ijJ&WGXe!M=peitbOR{e$DOh62NKi1(<2Vu4|(GsI^AwsjC"
        "5Hmh#PK9K)gMy-D9AERA%Pwkn#c{E?oK70==>?0Qm(%O3>%}=7-3MV-msi7c(?gS&S0JdtX)XTw0(n@@R?~_1+H-<joGdSv)9Km7"
        "J9Y!i7Ss9Z;%qkmsWy|5)6PJSL+z<9+8PSaO)L~}mWek$_kK;j{xyUz|A(>q4Sw)OLm2mK%?7Cn9<Gz=Cm$^o&d)wx)Et<egJCuN"
        "c&WU9tiq33k7tRtt;DOL)O?_Z*y~^($a&t;tZxS0>G<HZ$vkC7B?i0ibeyVR!>$$~+YccdO4;Cu6Zk(c8`Bb0%%Y-_WhHgvzzvCU"
        "@Zh{|-%Va0I{<>dJ?oa>A8Z5GpZ~VqHP8#JKQxvV#ZE_Um%1Nc_ucj!ss__+sp}<^cO3>;BX66#;T<n3z>ACYEKs6c3hb2mdr;U9"
        ">34*j-K9V5!)}PKJl!w{%*1rsT~lk%!~t+wv?fwFs>f}lc<##q3Iij`XZ`yjba7<~YCm=~Mb}5e#?5|U(EhJ%$ux0Xo)|wS6HGde"
        "sbL5GH0M0znEi6a1%eJBCLU)Hb7JNKsa?Bw|IvfoZ(a?0^(S_)x_-QxT}~H2uNogV4gI={t{T>c!;WJZW=yR4+%DSxAeP{rpUK^-"
        "{I=qQD$enA-Skz{yav<J#;QuyeIqtruTGXr4<nAj%s>#t!wU~9bHcQDshULdLY@~@o~ZqMT?)W5#v4!fm<AwvqG}nj#nGXxy1QSv"
        "Tk?Iu@86o$cL9;r2S<!>kfQk=OKk0a{*{btGyO_&-<xo#FLGa<xnQuCNKYghN;esL=)At4b(9_&XY^~c0Mq+^4W|pSVaXdWxH<af"
        "`1p9-4>TR6IbZ_os<_(12`c@aMY+D8&})Tne#azBg}mvD^}V_w#S6anRy1R||E!vUmXtevLxb~!y30gxGY}lSzf3CpLk&1U*k)jn"
        "5VaXli58>FV;j4kZM+Y*Y^K8Ag*;ZkO26yo{A{sp@Tkl6kK@5~ugu~QE@QOM?e@K96W+Mpkfl4%qNP~@K>c7t>D|JVei%T2lMx5e"
        "Hh|=nB}Pzc`=X+g+55yASGL{D<{sn)!T5tVsXs`?iFjRQSrZSsLLzVK7`o48S2XpXXY6XOlq$o{-)(5U*($S{?gTl<sFQac9hd-u"
        "&#vH8^U+2T^R&lGucv|cG8z(K1jKKa_|FZV{^}3b@((tDW**qP(SI#u%sX(~a2-+Lu>T)WO9KQH000080000X0JanEsMi4i02TxQ"
        "04e|g0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFK}{YWNBk`X>V?GE^v8$Qq6ALFbuvM@H-GX*;rx>I}IMP(^6o=&=u>j9lFAh"
        "B|2p*O9mw`1-icbEIVzSEG<tW`TbB5g%Inv?@Dk@LyDw<%U|ndjaJ>jm&<SGUtvp(&^T^o_y89FAOl%tdX2DgMuX%)ja0!GLWnd;"
        "6KXrh0dBpsY0|K3K{;bk1#;FGvR1G|d=AL6Hn>A|GTa!dpSl-A8s)}bO^s91Y_M%*p_~mI9iO1zv6I9XjG9K5NJ|rq4Q!bLR(^})"
        "jjHY$;<MSgd>-ZMSn7H)&Q*dB41dR=a#((Zb7!$k6BvTs*24;}uBX%^+9CBAvV5uxj|Y3V;q_qdJz-){GiwxNeUMgRMLJTV><l{^"
        "OY#UmrRj0a>^N)0r@k|!WI&kFOL#b~`zZ3gVl*hE*4b(73>hz6$gV@Hv%`tvpBK7sJAc%O@$i2shM1X%@`V>zyufXpMpIeYV&%As"
        "KXoXv@4v1vFx3%tES+FIo~WVqJ!2)kB3hkE<B#oTueUn}$r}vAqQsTT;^YU7!Jg5EqP^3^*_8Oz5AJaRcuzi5?ruyES<}7;ybqc9"
        "ZAEeG$Y$}sa(Dw{DQ?-gu{cZ#Lga&<RZf%LtVAE0<@>l4{{T=+0|XQR000O8001EX6y<DOKmY&$LjV8(CIA2cb8=%ZVrgb?V_|G("
        "Wo>O?FKKROWpZV1V`VR2UukY>bYEXCaCuWwQt}MZO)O6=N>wN*%1<dxF3HT#Q^?FqOD#&xOHNfNO3h8o%qvz%1PPX;rWYlG6(r^4"
        "Cui#^DJk)C0RT`-0|XQR000O8001EXDv?LZ;0FKz<Qo70C;$Keb8=%ZVrgb?V_|G(Wo>O?FKKROWpZV1V`VRAWo~71VRUJ4ZZ2?n"
        "?OENA+cpw^Uts?SR!>r@2zJpHSLYPmOIj2M1c$WOmu(=>5@i!Bkv@{z?DpFKy)*ofC|Pzk#a)2|&UsjyAM-uGnPHM7`F^t{`Y4ql"
        "2gO+PL=<b8Vxnf`zywp&a4B99A&FuANh;crzyCyPS|1sainR2e5nW5gbde;<?al2?t7K0aYG~b2t(hhq@0n={5BW7CRec;U?RhI}"
        "+8tQm)1sE5<%e+j&EttF-ZNo-II;S|d824;Mt<+uiE{Pkk@lyK<^JbC6l-{mI<3(_5ep;lnc%-z#RIKYs-Bt(%px16D&)l;O-?#V"
        "o9Y21YdLEwf0L;-+PbD~E4wC;{FYHODE3yGcZ@bqZ<LY>EpdDEx6x85kokp)@<%nW?Dod4k?*XZz8jshyS*XsvCJ#p=p8WwXne;7"
        "2v`)wd$+HRJXMNO4L9J2a;rfjdmFC8CAZNqw{e1a;E(ft{MWG7M@;38rtV%3U%Pxxd`&2S#FM{r;)Va))5l7%53wYHjnu0INJ-Ld"
        "PPUt`bMn<Di|80>XjcUlBZ_e4%gr}m!_kbqe0|B(5a@5LYD8k5P|-lv)UrPfhCyl!8j{2lClIz^ZFYJ^^uP_>cMOAxK*qooBN#aJ"
        "#5-wx`XG7U5^v0Ny~$t6cG?3%LYl~kv#pMablS?u=Sy0zFB-$q$#xSxMY(3=C)y3nhH9Ej@KEiC=D;959&4}{Ym&^}dw}TQ4;@uc"
        "U{@O$g=gclyRauJ4LLKa-gA+rFQU-N$%{yf3@org?LhSF=U$hQ09Q2`^A>MbjjX_^Rm)9fWv(VFK`|85s^i%cuE|>|Tqwj*e~s&&"
        "YSwYVs+tO92=T!91FPmK8CW*#fkUj7L;&!lj{Au!+B3tHPP43VCq@C?&%!*N;8sZzI7pM{I#jWL4ID7N0@MR^qa9eMsXj1%IGRx;"
        "u=WF&gSKWGJq#es1s8`14hr@_yVT#3gMmDO<98mI945g+0s(t4$ExE!H$E54%yl2QIl7uraN*08<(|f&>Y)cHx}yfn2>RVcj(MMx"
        "1}#&%6P*-?+*_=4f3h=rU<-Xe5(RH-ko2CVur;T)m$F-2*9PpUM;kdwLK$%s0*7ohx-iXcgs~31hTrB4E=A<REDKf6>1c4gE&e6x"
        "xHd6<%nu@nRREa~N8-TJIkGIhVI^$znMVS*Pj-h5t#x-;B>0Fs@Z*pO2mo?XW0cFOoFs0e!aS=wdEgN6`3$}M>IzM_0W^I{w6dU_"
        "xsQOz&3C(;yxhKDF;r{4jbf)h)^!UhWmxJN7qo+PS~D_!&J$Z8*dl={N^!5+iWb&~paj%I&#|EDfOz|In-xH&2j=J3Tw)jWbYh}O"
        "Es>Q|pq--+p%V)C1wBSwFkp&mF9*>e22Uga*1p=fj7Wf1vQB3YgznF`LhP*x4Aue@$H)_5TROwNEPxm^bK<OFNNr1!I@4si*D1H9"
        "IfA1sUl!|Z>avt=T%i$z2E*W{Zw@px!VpiPznzMhZO)+@=EMVx?f1y94nSPi=eG3EOHGNb4|G0Lm-CiEZ;L`h?iz>zsPQLi>f<Z$"
        "9)fsJpP=7SP~R!o6U&id9sC#?h!SjyjN*v>L6E7(@qIuJ7*by9qEpnc7*8$~91SmE7nqe7n)K9L$l|JB8|$@=9lv<ivts`^6?`j<"
        "-;DypuRKGM@#MN9=geK~_%(k$4f^JUHcf?up*jZ#+BqSj2VyTH%{P6EzJHz`=Liyxf4t|zj|){a-<)pp&Si-d7$js6)=s41ZA;pr"
        "Yk3F#(192jExCY@9qgb+1|3#IaSP$}vq;X$VC3RZG#91Z;>8tF%#p&x#~-0!n<J3nS`X~GQed0yBd)L)eOLk0-G9qMAi<wt3FZv3"
        "WQGcdp!7W(G7U9QA-;A`RSl#nd_w2|BO<Nu{+%e<v1CV*K{ExVHnDOG153_KzlBlI68FhdeoI06XamzB3Qu>{W<(w*7*1dnxK-SP"
        "A!kIUt`;;N(=TY@)6b3lE2(EGWfNRXa~TdR?%UnwJ^5@h%pS^LnA)E`X~CV7?07xSBEIs>1|i2T51gWj-BDr=#-l>ATfDG@(b`4w"
        "Zt<A%v59<~*|BPtw%$aK6chBtTUTn;#BSxeVj9`Qp=;CqU$)<T5vKd)i_mU9+>J#7{I`&MrMq@sTYJ=5rug@pEk^kd1Lg%}92N~o"
        "H~|iRV4gM3k)e)E%-yqF5hveb9z|DTh*_6DGo;**i#MvC>mCCm#FiaQ7QqBYyws6|V2TrN7BMR!5bgK~D8uX8{0(H$tDG;*(>O#r"
        "HU?|&RAYF~&><Gx>wW1Zrkwv=oLh^*^3uj!a##FE`HS&kZ}c^uzzBiAcQ8U=D;+;{MUHp0@Z4^kHEilf_#DqF`ktTGk!=Px%gjl~"
        "iU&oB;!w_^L9tzrpglC3mS?ttLk*_~A+_OkWt+9AwLO1Nj7r`dG}h7O>H}iA-F)%oMPw<T0V!p`lHGr}rFCx?e}3zEspTAByFaPp"
        "{DYll6cBdCs3a)WcJjk&gBiDppOfTg@lPO!t%^o;MetlUYsvL~?2I7vI`CySBK*I<`CkXuNX!+Sjj+;(^vWR%z-yOR068)OcH9@C"
        "eGamphR=CKHu|rg+ne73P)h>@6aWAK2mk;8Apmvagc~UV0086y001KZ0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1z?CX>MtBUtcb8"
        "d3}*hZo?oDh4+?x2gEwc4Z29O$f{D6c2`vh!^EZn1`Xq;zJ0M%JN;4225H{oW8RFEa_`P)v_C=!T4*wzV=&y4_9CF|J>-orJrP9q"
        "gJ8%ER$Lr+A`g)}U}s7-W~988Qr1;noe6_zuH1IyZVL%~-ZtcH3&Mqj#Nwz6>Oh$DjthYu#{FbiA((QhmUQY=B785MBF((r^UHBm"
        "o@QZ%1EB$HU2OCGTBqa(bwRRf5miF(%&@`<KrL9M=KwS91ZOBCGw?VWmVhg$sM&oi!)bcwt-96}yEv%Hw(_2^FTVEHLBaz@hEF~f"
        "1!yCt+m6Csv+2fFDW$zvN<4_yx)SBcJ0Hu<w|;kPi{oDQ#jrB!#q)2OKk$Eo-IU+?#`;@beF0EQ0|XQR000O8001EXIFgjua|Zwb"
        "J{JH0ApigXb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVVsdF@XJsyMd97GmZ`(E$eh-lUKx8jcD=U|^d+^`^y=5)BbzPca1BybRCCcWk"
        "TSUo8`mo==?~syA(Q?*qGlJG455IGGF5fxw$;rttRU^xSXH1ET*SrzQj5m!aXqD9>6}2d0!P2s3e3@}tRsk!@hBa%!5|Q#YYZ$NN"
        "wQNM(w6zG2kB*P7X;ao%nSHptW(}`bqG7EPisb^|6SmWm)@9CQAsfjv`B|%Ao)Q2kV|DuM>c#6f*RNh&|915%%oAV3zv-ocw_*!`"
        "@nY?rg=bIwfT<We#Y!6(*QHXU8!4HzxbE_vk$7lKl||JyyG^NBTf}Q#z|il3gXg#uX)0q09NH!aa8`*r#sXgnR;H}Nc~Ngga(tAC"
        "xJ*RNmSWAf628M9{HWJfHBSgeh3rZ|*^eU6IV@-dcC?WlzRask<d75z%se?cp_^l|tQ(f|W_^65J=fe<blYyDkEbQul!(swDJ%-c"
        "p{3Yc4pZE`O9@C^63+2aoN=Yt)z2T^m38hx>gO_PGvUvVj~G6%4SO)U&UYb1uoH5#cHgp=$x3|R`E4s|nM!P%yJ!3!eyK&3@mOCt"
        ">@G{#NHG>glpaxptjN*;iIvaEVF44BnzJ-3d4nW47qjnri|C_T#K-f);k0k<0Vu?29K*lP`mK<wH6h}n?@<)FXaHn9R2D*cFu?b1"
        "15qBgyiPj%?&h64uSKEC+Gn#LMhljWTV6{}1-4*!VO8FG7k-!vUU;+ffNg}RplWelx5Bc$7AUiV-LXd<YOQYS#v7m>54ZCnV%^A&"
        "<&RYAFQ_`6^~LdA$7nLTTAFF`*&xr%0hC5jpwKA^pG#BpCVyUa8P5xx<*mZfOH&wO@4RRAN?Gpc>L$_=el3$k6clZEjr5_$bWIzm"
        "zQ#NeuEL;8r6=swljp<{Jz`gN4<lA}S(TKoeT}-97bDD~&7-y833BgUn=c^`oE^vLElZr9jsBN|Vq~UX580cPy+cw3J(ns3)%F8B"
        "J>4m^P!qDbZRTtjk8X*sv*b&e$>!cV&@pVZLM)>#&*rRSQAXs+XqlxvMoNVUli_t#F0mx@EP`c!>9w=7Cw{O)*TrKIFNdd-9ivhT"
        "@GL8DX@6iqD&i8HP<$43*>MC}|EEbI`?~bGCMHQrVd!zyqBKE{(8W@D2W<!Fc*Mdx_T6EFsiK=rMD#Uhzpn-CuPF+=s<Qi@^%xxd"
        "tG>ru0HWZjKQ%L0o3wEg4y7{5TVsL=4kSEL+JVd+Z&?Y54Whp9f~kD#o54p&eS(-yw22hz85D619!eNX?rCtqf6KDWAVQqgiD>>K"
        ")nCKSb_FbKH@O^M*mBmBBR9B=XubdL$n=hb$naIB)_0S5bQLBtB@ULUgDb(0y@if3kH(svgH%w|8qUKrcB*@RsHuK*#`T;P4IH(e"
        "`pl%yH|+hc%p8E5h)U43zn^3wc<osi4V*g#Y$>@~Tyi+XHD$ApWJTy}khcFD5)ytN%RN52?+D9w;jeFJ@3dcsib?lEFn1`@IJgT="
        "KTsWtr{%~mJV9yspi12NO+x*mx%VE_<On)mM7JxVzTar!m*&Ci79Cg70Kuw$mYVOzebBRN<28B)m_;`kq8lHWF8$bWrXAh+nv0&I"
        "in7A<&N2C5>6GEy4Bz}Ql|`#TT31q5u5bpG^>|N}0m0U6f6L7$-8Uyk#YPW%-bbVS6S4}Ux>%S^O{L#nOplTO*e+B9t!g}GPPC3M"
        "6wA5VSk1zgcDlwxqbz#grLiA@#)L_)jPC1~Y4M?*8_oncbxWg)db(QkN-%H9n|Q6OG|EIVDp%i`8C+K!!h)Q^vS{TH*4SUV<6&f{"
        "=>k&^(q$_ciR>#m*MFTVr~2pv1jinq-C`IKx}Bb$BO4PL(?#&&kjVt0*a$r)?Xa%&+kxv+Gr+f1qW|wl=iVt_=T<%B^bbo*KC(Ks"
        "fetnDPCBw29a;T)EWMMz={u(BN!}RU^5CWyG%d#)16zXIUT}Lr(59IyVLXFbuy@YTx~b>fn43)SFd2RmZl$>OY!8jQe=u(Te%N53"
        "RrDs}Ax$ov>um<O!G-@8j-0Ul?{K!Op6OZnUvxw(#I)<(ICWt~R_||I#RIvh7(=-=%;+9v7J*?{3(TLlnRkPiYwJGrvONPA_v4tJ"
        "yKFVMhF$#jB<NhnkqqHE4{d6}F6R67{tf#uMM9cozmzjZb}dGaALl{$ZfX9VfBFYosXvje##h8^6Go5SYCVMnB9bVTS?2v^BM0Wq"
        "!8@7BB0ce)TTIL>59&d~w!!T<X({t#2OYW$Rhc$9zw@l)`uC&7?A$+iY;VHxYyl_#e%GsdjKSN_1WgRZKS2{3*Njl3{IvB2Y<qx~"
        "mOS3r+%#w!CfR9K{a3f5!?;8xt=h6xcr}|QuPNy6jUAyk_@g=7>$Ocxgw_uP{f|gxF=wBF@du)8jf0O^|65hz{SmO&mUcav>7Qo@"
        "FY1HCbOz=soz7uWeg8MLu@p?Oz#on8d3^LYP)h>@6aWAK2mk;8Api+!qYA$P002V+001BW0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`"
        "b1!3IV`ybAaCwbWOK;mS488}*e;|BnfyUWs2OowFLkkQ;cgTWW4Mn13A+jitOgit^kB*zAD>`7h2o@jTgLG9@kMG_$P?*BeYorX`"
        "`5xsCLKKbJD|Ac)XpcZCKuqZ<%1+}6K`?czuRj1Sb8t#rS5>uKEEngJ;MDo#L{K0Og;=joQ_}OBmHGZp(Jpr6UE1R?xLUmHx-;sW"
        "`hZT4jNeE%=)!o<OgvRC7aw1AK3{O3Jq6@P9Eitexd1z!vf@u|yO{ULg=5wROwAm{xuP}s{!qw*E#oPQnerkl;QwUxF$B!5s8f_?"
        ")>v;b%Z2bva)X5=T6Fw8x!znLI@qJO#O=)KW-|KjpY(5G_2Esu)3J#>94FTx<__5!+8}cp8#v%-El<cc9+4xiCA%U$>00T4IsQUk"
        "KVjVWD(g?*w@@fWKh_ibOY;q`W`^hez^u`5o@s*X6d5o?l^V9}Vm99MdV%vyrk{!}&IELADT+nF<zOsv)l)IvCzqlPc-l^90GZUV"
        "+%92&-DuEJ&Y<kP)_*C#ei*#qn+tR^{Q=b-R9}XYa4-?C5RJ;@|9Ylu>rPMEOzUH*tJ~$`H&9Ch1QY-O00;m803iU#MspTb1ONb~"
        "3jhEv0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MTXk}w-aBpdDbYFCGVQzC~WpXZXd5u=zkK;5Dey`O3VK6TV*A4K%OGJ5r10exg"
        ">9qIKT}5s(X-yM5vfb`2OaFIf9H&j%9(#U?G9J&*Z$4WH@xfbFdhqzk-x=M!fRi<P)XrE3-b(Fiv``xhev8mqB$?7!fwbjTc`W_j"
        "vfQjOo6qL6kFrsf^yok~4eap6ee1TeqyIIM6$GhYpfue}27ejQ=%0<<K}pIElqM(2L2Igg!{&q#Y*V$Ju^ueWXSFpgbkc7dwT39a"
        "r{9s9Hxyaa948aESyY$~H6NMJDy&aw7F>m*++w-wjM6@t&w$oimb<mlxCoB(;BNunEWqtlS5W+w%VBLy155Z-HjWi7){WY1eZe)I"
        "?#^c^-2DU<l|Anru(tX{`+_pZCTpbL?BxcF{#W!`UfK<M(i-B)oTow;Y$!>49sJZ$N2ohXc<m*;s0%q9WV6S!j5|v5MK2Gf5B4+a"
        "Y3l}iyi^U^cu3CKQV%OW3`+_b+cj2&H#;hG_Kcjgh^LSJ=UMsfx$KGaHsM013#hGZkpZw2ofiwJ@SsXuin7~_6r`h;%L;9N4n`iN"
        "YUH{>%Fv-tdWDQ}mZYPgTjiY68`4_E>b?sR$Fw=$Q>`e$i4JE=`zKdVlUhW}qnLQnP&kB-T+8>?8k^MO&ltiPPY|J)!)vs%%71t|"
        "iuC{5GK4ua=E89pE=n^Nf;w2ZhKiVsQ%ZX|tByMxGxGbc!6fXa@YOQ@hXAr(dg1eC!q+MxAMqgsNcpzXm)nrwD}{8-!I`}+5nh9j"
        "<6;2#>$U%*xVC<c(CN?xkP_KML&_B`dKvs**EUEPL2tEH{H~Q(#Bu*1q1Ab%rB_CSBi5mb9Y}86^GGh)xXo&#Nv-0R*f6cD5y*JZ"
        "x_*XFO}sf~N6vgSmPqUprs4uneW8np`xCD7D<p6g&&td%);k{Gh^7>-ti@n&Afz;!$}{F1UV%17HdYN$!eBjrx(exHSijg$k2ao*"
        "tll?G_`QKqD;oKN);(s)WA!e5%p5j#A-&fm*%frJ{r&Hu|Lh9tSxbNXA<fu?1&25_d@*9Z2a1KzO_ptPzEGlfGBRPWnEQ=*Lmt;+"
        "#yVeEVSDxcQwl!<7q&N_-agQ8t|sT6&IRC+*&QN8i=cdw=8a}dft^XhhOQE(ps6lH^-EZ#77x0WVI25!jtrXj6e8Z@fC~@dA_~ug"
        "Oq*h3WpzcTyiOcE9`LuBcvz9vCY0QVJ>}%vr@w|@lqEOt^ZmnJOR0yq!YB}aJJAdq#rH5xJHek@Nz@6y-9NznAMaz4vfB<uwlcO="
        "Bz?d!9(v&ItscJ~M0geTp^H3Tm5a%KxYY`>rm!NF?}jUN#KwV}7?ML3(_1DE@%6}7KC3*n978$_#;T}GdP3QrAB)IUbmB~!W9=z;"
        "8T>IWK0$9stdgn9BrHul#FZLyiK={d-zE&s&v5GKO>*Mq!7@Fe{sT};0|XQR000O8001EX`6tF3a037Ug9!itF8}}lb8=%ZVrgb?"
        "V_|G(Wo>O?FKusRWo&aVV{c?-UuSY*aA;p}a&Ky7V{~tFE^v93Rc&w6FcAJeY5(C=B$S9WbTB4VR3sQMkYFI#H-sWLxweMH4z>g9"
        "zvukY#0dpNz9hEqo_l<E&t*27J^OS6ci~c22%n7VO4BKX4@!Q;(r_u@ovgYVVIvj1m0~-;l)~Vx@gA45X%+5JXfn-DPfkyQrv}}%"
        "ZOyR)8$eiNvu_dJywJcz1$r+GyM+o{vjZ-Sq+P&fja9|P7Ie!>B)}VmIM>`5R6tP`bEGoshoVAJ?g6VU>db*&Vbd^+hFirPdq$TL"
        "(g1JT8XKgOicn{>ndQ1svazgeKr)cZ!28y^vDzMoEhAOA8?SOf$Qp#OW~V1*&9t`nyi%;)g$ziQ2$B2jrOVS3poilABR~I@91r?T"
        "ZFRu6kLa4h!W9l~3D+I@DUchexHeprW*RZ^d5ALoLzZma)lBWvd_Zp4ZQ&>_;T^l>P1ks$RFcxU2f{?xagAKvx~O=wj9ZJ=zH^>{"
        "Mw?nVd-U+awc5)E&w*YK9$hT_`uurZ*9f{&vXz}#!VCWat&(fD<~29_yifZfSru-eD7fIJD3Sq0W4+1#oI{EXD@!Ijnhk$^ah@H$"
        "aByCnKhBP@-qe!W^F_W0$0^L8!nG6_6_uI=Rg&gC^>oZVA2TMr@y5iu=T3(G_r#{Z0onB5hcw#f4qP%Sh*Z2}^+e@;uTE(m7!}UK"
        "@)Y^YgK;t@oego*ap2b+6(b$G$o?XOIuc#V7VitR-M)nye_%aLm-9K!q0h&k!@uPMnX-*`=;Ts1YpRUZrkAc$vE>09WOk%ZOJJRS"
        "Xpffk!J@bstU1%GSFF}Jy+Y0Qs3=>i2{3kjP0JBdt5;_@zxeI`K`YKP2qy`fX{;5}-cP2gzepvLiJy%(q-m+*Uboq#<HKzI9NT>%"
        "4z}6j=8w-K5s-}S|E4q4;5z)OLg)3>*KS%UY$c5{srFPE+mrGef#q=(CsT8J6(8JA&GD~`HuR3sB3Uz2Qjci-UeqY!IeF~PU<Sa@"
        "n7(iS1v<cfM7pn-k`CjP`Z!9-Aci6A(Fk#m=c7yyi$eobp^j-d8PYbA3EzYy0N3spJQ->Bp&w_{+VI|R6)Nwx6Tg|kBAX7aIh>_q"
        "ponAtdG9x=C*yb{x^17w)gaL10G9s6a1dhBpGEe{m2D)uk=-v)O9KQH000080000X0P5!>`^W+S03`<i03`qb0CRF<FJftCZ)0I>"
        "XJu_|VJ~fOWMyn~FJ*0FWn^S&ZfA2YaCwzfQE$^Q5Pq-J|8V4m7PZ;LW6=<3R03&3sOxw^sB-P=VsPwmc2=@q&pAocrE53QmsGy<"
        "ci;Ej`I02LK7Ts{NzVJJba~+{+5lTe3PPpr-8=YpH3uI=<-ubiNCZsb1`(8#{_Nx8^7EJF{Bn7_n5Vf)lO!1qhJy{c9Fzz`8sR;9"
        "(2PuBe-xA2v*1W(skI<HSQ`Vw!G|t!#83}tFP2o|csQufV4km0DQ5T087->?`qG5iZ~*)Z8r2G|bP8#uu>Q_q?VM@iUfe%Qh=Otq"
        "NPY&|hBhS=#5vxTByq;#sbQsX1EMHQwTk|%aK~a*)HFuT?G}2*V$zQ2q=|KWPRCXzUu7s%opYrv3ThMHOge?f1#)~0)A!HX!wp($"
        "gibfO^)Hmc!x}e^5TvuLx0HdeL&y+Z5p=E}kkV*Dg!t6d1nuhtVP)nhg=NME%5k%;3z1|!IgVoQ4d!vIWQOv0;n-*pMhoBI5qh2h"
        "5wx=@Tnb|tJ1>2JjWfn=Vg9#3(h=aMK>3>Iu*#V`X*qS)RM5DecKVt>dUNP#x<!4Lg_PS{AiTQR#e3(iO*Jo}J;HZkN}Llp8YMgB"
        "SJw~$8<rUl6#Te%QmjiOsG96>VXIZhenO3#CgYR5)@(&|Gy3+GAK<eb7Pvv;_G6#L8X>|=H-x=f$A)DO&k1etf6X~zkR~3cN9%)|"
        "`={hX5BwDd_BQd{D9E%@Xd9zC)-AD7zt{m!h4)$>BiaP|cX<d8uZQZW%=&zPdHkZ{n|fD3w+npJj*`>j<M8~-pye6`*Z4i9^*(}&"
        "LxQ%<(Tv7VJG2{=K5;<970>su2bT_owyTOS&7M*@e`Y{p`99Q(7TwyoliO#?10hkzI)XR8As*nWUAVhSXS?^)Eve41<1hMzI>Jzr"
        "J-!V`N8ZJxF<=78u8a<`@IO#X0|XQR000O8001EXv;=<25(5AL0tf&ADF6Tfb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVX>MtBX<=+>"
        "dSP^FZ*DGdd8JiLZ`&{oz6Z#E;CxDf)ahZnWauzpNQa_pk#$9{h9c34Hd!(#rCo;o_)&f*OS%H{$snnZua9JsB)1oD=O8yIf0Rz!"
        "042F6w8BaRVHM=tuq6a5f__4HM!nty2qdh}k<uhd#-s5lCs%+HK}aLKM-RFv9fe*LjT%{SBsZzGAUs&xy<X|uNnzGl6e5+*=6c<t"
        "EPj?qx<DJgl_-zddnxkV8P&eHLJ_0BbKxF^+AWA2MNi|=$1bD^!~8;f`9NDd8;>d(Tw5KqF!~p#>+I49J0FjLKkQ<dgAddQt#k@0"
        "TRFAMyxWq*N(;Z?&RN_ydUmAzKdnlGCv6ngHUoXF@Ebv>Di0YqZsX_$2jsq^jM)tUL<pQ2%d#M7jctzm1^cT;s6IKax8<k4DHHSI"
        "YL765Q(MD?D9zT#A{s6AkYiKeUi)o<3RXMFg=Bi{;P&EOORYidbLLBWTLB)c%RqfP<>Lr_Jo6(Is0*PmXHQ;|USX#9j3%CWIX~d("
        "8JvH3flqeWOSpsxkv7tltA-}i%;$Vh0cbp)WJHV7NIR6K$9`877)Z+lZ=iV)U8rEIk>3l&bD!bO0df@9sD#{YdnC_Et*_`ApITAi"
        "w3qGYKTsT`)D|FU%-T1tNi|V*NlwDO($9Yf5Q{eeNw}byC?nm)%3#uK4m--l@B3Lr9N{L3#pleRAj^Z@6F7<Q2c4{p6RqO(hFx+t"
        "J*av=*pUp4_N*s#$75tC3@LQ<C%dtJ1+!n;0`q?4P2AQqSB7(K+cRREP};7i?XvhfyO}*!6l7pN4+U3{^=Z2YmV_RO4L`@N&)oA`"
        "Z5h=%DrWa?xtJ}#EM_;4=W!wO!?;^GCmDPS56#JH`wQ0(-Rw8T<+EjjgRnu!wG{>m<Ymag@WIl7ZQ@>2n)uJ`-t){rK*5)jqMs)u"
        "&lqV9de2TMreS~^X?BA`D<}o=Fa&;3k6X39A(CmZh|;c<&7cX*tv+_xCh>$PQ%LUWDh_UWd#7O|>KR`lTHsK1u2H24GvN<VO9KQH"
        "000080000X0LG+}9mfU$0E-p?04o3h0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FKuOVXJuqxVRUq5ZggpHZZ2?n%~<Vk<1`rmK575M"
        "LlG!}G}}&q(4a^Y3^?vK-ECuErm1ohcQsAy)ONb-3Guh{*iITdO|N$nAE-2~iJ#Zs^XiU9qpQ=ibKrGJCCf+w`2UIKW1=<Xnh6e)"
        "&Z(q4qX}@KfxvIHED023LnR(o1<7cMwCVBD@zEcGaSck$bxGa<;W?-Uku--_C-+cE@fXdEPzCrBSXmX8mF8(9^+vq8OyJGy%jEco"
        "6V28XUfqw4491^MGLkK*vSRT|wydl+E@Vjx_MTXCOHx%xi<Xsyakx7D=H%-1Tm1VD4ZNika*?T%i<_T+e|dNL^WC2}muZ>9Tu9Ih"
        "3iKVxv@tZVt)ZX^!lA9of|X2LB??82MkC`e`XHnREo8O`pHj|&DB#?X9UtX%4zrpS`SdSi^>ob?o3VoFO>}$&_(6z8PUUoszU1eP"
        "j62Gekcp$n=^+Y5i~01K@mxHggTdx+xE7qk#QsB~7{)&ZjOUIKy?I49E-Vj?o}@MD@rPD5WTaN4IB_ayz9oUDLd-}}Y{2?e(A45%"
        "M6-G5TFP@)!e<kB0SJ2FQ+L#uWj>NHMd6+lHNBKl$Y^wLXc0J!wOT4DYo%d9FjXFAL}!ZxVOZ5BV;@cSRcK4R#}s?~aQbL;hL*G^"
        "A2j5$ao%J{i;y(|6nl?}nzK39j<4Pe5HMv6$_B_*ig)(o1WrGw`WA`3B5l|#6f+8yP|Ot1R&_!7lU~5EmuHWT$Z?tKoHRK?B;ijq"
        "a@+|ruUSSXp$zx3(bHu$%Gk<#u-OOakH)X=?RjFIe>{Ya2ZSg8%RvP{4akSFFWIY<CRfqaPVR$a`Lc)9Oo$?iTaiYUlaooIlqV?6"
        "uU)Q`zN;|U(fBNZ7S{1ugPx?imK>taUl)dq=qU0nPC~sghK7|Bl+(l1d=7dNaKx@#Wd%_RZrav=WrxB=n{E;3=@qs})ZF!tTWrEb"
        "H!WaRh-`_8iD~!S+Y8r0DUlq90a2T>q#6efELfgX?u~}DX{@f{&0#uaoayP*1H~YbbdN3M2!U;S-cjrMz!6To>4K76xkfpG?|X;K"
        "b$EW}otuNL&>ru>9&BOua$40Dl~J5-J&n6FEQ*rx$a69Aok}`4lECO>Yhv8*CByw>8(Wh}I2mjk*xODT)3(atix9Fy+TPYwZosj&"
        "N^;g#(QJc5-<<{;4YH{Rg;#F))T8rI^4i`mUFXB3^Jjpj^{pXmAA6K7>PXdo?NRtWZnE2#cKqxr1IeQhSS-Vrxd|IC5pj6rjXG=2"
        "9tg~b78o5A^_p?4&ky9<j1e{Yl_0sNtRJ71wb-Dvl{lQR@YNQgJ)*6=<wwZd#^&QhuK#<HqSm`Yhh04kX@jHLYn{BCtJ`aQh&my("
        "Qj;?}{%#koPvrU>{=}vONNacCwJB2U9=2_B0UB$U6H}B5F9*1?<3_x|(Y$T*X+OO!8FtI`)bHXb`<cjme-m#}QMh4KHMrc`T;-oc"
        "wtrLhb?&Fyw=SDcvumP|DGu@kre2$I=DGDM=$O3%4&4dXQ7b;A_SWTqVSHve`fYEAc~G0rJlIX+gWder$aAYU?!3b8ZKYj<$-q?U"
        "4-sACEif9`%VT3#cp6xK{^Bt<ZaI?B$bmK2^QaR%h^G;J(CMK|h;!Q9BXOoz1;bUx3T`&N9oL)OG2Qv_g#M+uV6>MTr!o^<n|m(N"
        "X1l@{X4TMQNqL~mkkNM<Ci{j<4`PoMx4P!#_F7=~(Ga?%RnhFCXP$)B?&R`DMNd#~j-_ixpQC=Qg_Ao8XK!Bu_Ast+g`pAd+%JYl"
        "*OG2rBs_O}w<Alo#}hZCqK@}NUNpLr-QD6paAdv5!u9{g!d07uZo);YIK<bRW6`qj**cvU0<$m1k?L|$s+H8;`&6xJA}M|;DdF~-"
        "wWPE%DS0Pr>?vP^oq%zw1#;zjQ~9I!Ns78strgt>KRMaOA~buMN?K8(af#PdVlA#|<fQEugmGp43G^5>^J@!o?fC^M=$=W~{gR!%"
        "(_b1gA-%%J23FQZv}(~r@V3Bqu{EHMU&JN`-OL}5>As|<LDGCWXGIbH)3}m=6FV9+J|D%wLUOSdEQf1y-EIwVas3O<!)$3E=5~^4"
        "CTZcCF9K5$yhY})veybJo>jWn#peYsE>UAfO890sN0NpoX;|J<ylhVkwxp3e@a@I2nh_fY<~!Z|1d77rJpi2UNkKa^a~tg-D1o8`"
        "*265l+HX(xSEj97)4SPIEo#-jM|@2GZS!EDD|la<A2)(^Icn`gx_<#sO9KQH000080000X03MT~bg=>e0D}eq03ZMW0CRF<FJftC"
        "Z)0I>XJu_|VJ~fOWMyn~FKusRWo#~Rd7V|;Zqq;zeUFrX82KrYjHwV8DF_}aT8h*ZL=ik7WUakpXT*Bf-Cege;^Ue1C3VuY0xZSx"
        "WiDsVnHfh>v_5*Vg4!C{<erR%Lge1q7MwCZfs@->QBEGT@N|vfNw+{%t*}DvMdv{d)<>_N_7+LqCQ%g4XY(1G8S8<g<(u8RL~9TZ"
        "v<_y4H5Fu3m|Un5t4buf(FK(-crXOC6w+PRm_K0eEyH%B%i2g(9~)mcK88!Ydd9zF;RB5O-YyqKVU!%5Pf_^B;)U^_QONehT4Mvw"
        "`7Bq$IXLc}-v{R|(L(ElFB*l*)qDngNGu>tNs~|0g+o=u;28UAua?m~9IoLvi)FHPTTb3_gd_xjH(i{7WlV!QXAEhe0B`+}j5JrC"
        "l60$q?=PsGu`vkmIX8=zR>J+<eY=DwZ+9L{(6OSYO<-m0gEP(x-;d*S{*T}B>Bno(2Fk#r7q&zX#)53Ms3;#mTF6Dd!M;7l$X1ZU"
        "gPboyYeq3d--z@2%{F&NlAm<lD%2P|iI!7ZBzCkYQ<8U!T`J?d+lPp{v*-ia1AI-41`Mg=iNL*B@Eb0S#sk*(L+;xCogj_wSd~yA"
        "(=JEAoj80Qe`RGz=e8eM)oTD5Dx*us;dypn{+}d+97Eo1YsK*C`~-xSaK45O`|e3o7<(hE99lICW4YQH(ys2w2p=kY^=LDM078n|"
        "Gp@66v&RiT*<$<0%1;}mpqu?|G$mR--;gC;hGpnHv5m#6hLMfc&u5|btRW@S+9K~O(kRoaTObFQLqK_N*9=_T$v>>?gc}FKZkq&W"
        "sTl`%)rO=*4bt3bWrPI0CCAHhgLX2EkJLWle|_{ku<&900(3fGGwh-r6$$%`grZ;l*C9g|oVz5T=cVtk$`pShh8CaC{s2%*0|XQR"
        "000O8001EXBM67!01p5Fq%!~jApigXb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVZeet3c4aPbd8HfMZrnEZJwX0}oqdr4S+(7?&AL$0"
        "Z8u#knzTU^^u;g)jl`Kzty@IN_O$4J`<`=1N+d;%-F5(HMk0B5?l&@<&F05!DDOD?53lQty<|h#99h{6eBZ-B+g!0eYl^mJ_q^O6"
        "1`#oy?Rn3#Vc-o;h*;CIf!AHzXBEq{{J`Vq&z?W~aNun0#L)AshVh4ii6P5xS$p3I_U_H=i1l2Q#h6t9^X;2=47OoEx9?uZ?5CzW"
        "vV!MrVQ%#NSt(#mS7kXbV)j0(`Pr`R@3X#O;*j;cU_bl<sOQyKaKX0iaA1O0I|7{dt$=qx3*aXJ!OdoaSeA(Qz<w17f7b(@@=aB4"
        "<C+gykqw!8#qM}7O3yr?qaFJk_EZyhWyOV>HXOUM+3S%!6N9ns(DwPkeu|rhWr8(&n*mT3KxPe_YbYeEmmh$q-?h!I+(*Wnx4^W@"
        "vo@}AK*W2RDW3qbR6c9S!SiPfJ~Oy~O=5OO{w?M6COh(84Q5?e9n-!Yvi_KMtw?#jRYM>*o(<{k9XyA~0YY$F?%F)72!}W)S~MoF"
        "A3ENbHEjOA<2h&uBK*Gmjjt>PTJvGz5Y7H+9J+Cc*n9r*68^r;hWrq*5A7{)08a5U@3OLg{gBmN<%No}2hdfOfn+pfx#~Oh*0)SC"
        "8koN4MLMak2CyB=sz`rr!N$_NQk2`WDu<&F5Rg<hAfve5{>pQ-(uuwfX-P?`Qvlz$-4U7k{8_<wtRag|B|ifJv|i2Zi`a{ZjQk2L"
        "q^(%OKJ+8J<qe5*diy=tB@eHp^uSoYAFGNLC=j#(!NA@as1`#b0izQgVhB{WV7p(j9SEcCk1VV1v!h_60Mv2Qn9L4AmOE`v!N%0n"
        "O~^h;*yi*2;#GXUQTvkyWu@Tk=Mk7$^Vfaf_JQ_YhPz_mBkr=Q1lfT<$o9ZNF<6F>BocPK36<6g!3Ut2tTxzKBA_wADMon#(+|OJ"
        "BP}NV9T^)}?LF^<@Q)Zhh=vFVj6iTgZUv@fg$M`|xQeVu(HREBZxC*>8He53t4$bQZ*V|~ZF4iRiG(%t;l3=uu_f%YOB_TbSON@t"
        "efjN%f!=ZU^3^<9ViJ!X@*|kv(-@GsO%XtHarr7*8%)6t(+aIT1s05hXY!#e3f@Rigq|~w@&Vo#B_JVZy<{(#Z>9lhvF}TFF7gsI"
        "GI7z_SM4^d(rNtS3uA&MBFz}k&y*rpLTVqO!lv7@D0`{Z0By44WRq3Z26?YcMsoefw&4o4a%9QEPseEEfKomLa(`<<jC(X`iS7vf"
        "p44X$Ed|_-_M^ktH)@46`Md(%V$u}elGB--cHnX3hE2bN$i;gwAn7r{tMm|5L00gh1_j2z2_Y3CHrUNqxM>dDg2|j^JDf)0m14FE"
        "rwABbkc!KO#^F>$3|i1YB^Snfh#DngiJ^lRXdve3B!sli#H}$P<3EHK4@3yL=`BB|lH~~s${2oO{8j=@E#%w<dJ=gVcTgoQ{=pY7"
        "Zh<KiL7`#PpiK^*=*XIOAicV>^;>$vUvQf32Pt=xac5=~D_Sl=iLsM;hMlZwuch$tG6AtcOc;24ivi|<MZ6w(40n5&6!Wc1LC-&q"
        "V2XnAhpsB~a+pj{VTX!JyFw&os2U>P*nkZhw=xf?c(Vup!%{FYi`E#FspFPzb>L%J;4D15h)Anr?P!cW@)nX>QKI>JG<u%*apVvs"
        "6A9;I+CYd3)bFlue+?|e!uYP__W|J(hD#v6^edpjXhH~FTO;5d2M3G3@iL4lIKp%u06~|wG7v7MAj6U-MjqJ_DrjvaJt~sonBq0T"
        "kd7RZVSZzn5;#r=6oyGqQCf3HJTMJmv8`s^t--q6Kf$^gjF)*t*k55CekO)UT4n_A9!2mpV43z!YPqx^si@2MUelK9etmJ{SwQHj"
        "=ro68`NEWISv=gt%~;R=i=wYs1F0=0S~b#{&eKx%lBAev@X3|A9C}Pbe-XIESXOfoFaxDVni^NQVZ@|5`PIjOL$4we`6s~ARxAJz"
        "Pj`^Ur_zIgi8{vjsOh4*&BktQ0x1zB!Uc@`%bUOo8j&3;lp7W3zo8<%FPoyhH-JliP8FoYfR@-Qx9{7r6UIW#!g;CU;8PZhyzME8"
        "mjbA>p&l#kPHiz_Bsc@+LsBa#f%*X`@xYQlVhmBhXs9@<;D3sD5!*qfFa(>kve|9IbxiILyyt;h6=%(n_P^!4L(yY32U;N4L~6-z"
        "98@N)%3B`D0TVo^`Zgmml_rj&-3@j2K)?%>ej;HJmY<>XYb7^g8a)BfdS0RZ{KWdEiFae867jfzVz(U!$L(2bBGx66nIw)mYgB77"
        "Svil-fla!^PH3WjT2u@80iKK%)`@|Q-#5w_TvMc`M7bZ^Q3N6Oh|pCma|mk(%Iyi2R(c@0DIhANpPTM~fP{}K`v7<ZKgP-|*>zSp"
        "_1++drxbCE-L+i?kU{C3x*qFyD9xb6oRY1!8$c)B6hDQstQ^vmcnk{H@%(Dhwej^D%)A#;QY@e237Dz&WJ&-AbrQau5#1k5+~!x{"
        "^gm#!o*WD6fFMc<{u8>}>AZ3gR-dqkTI=pHwN|8QI$j>Vv$5D7oC-^?Lh43<$qgN>jXK&-A*q*Y*(A;$I$~&MtXCp1Bel@bJUE5M"
        "X;eJv4CwZM3$$Q}BcUo$^tOp&1F;Gwz9}1?^}#eFQf&sdEi)mjdq)D!Blff)0PzPfQVe|X0Q-$lRbivR;x&zn@M?*%T$xb2H&|(z"
        "%1PmJ@m%*LD^<Nm;*mu96gX-`$;d8y5QIRCz^L;O8$%dDh8})NrCp(LUW*Cz+^4ssT-B5dN<Xf|qh#XK+_~|zA~RZ)g6mE1+rAJ%"
        "XgQ_%J<hXSje$%o4>Mfjp$aV>ut6-;lqXpQ)t{5QiV+8`+d9!9oZNKFv>4CcPeRY2ok|$xj&}{7ujFGdc(Uqr<pOi6`aaEWJ!Hfy"
        "6AxF-RD76TSgI9a#aHf|#OgOo=qlc7bQf@!5arhdQ-1v^V&txxoNsWCGOzh?XbXeWx~3NaW-~U{1cu%N#?2lO&&OJfH6&6YJ7X8{"
        "TxE5a)@2i%Tgod{0b>evh$Y{s_X!0SBsF&T%V{6Ex~ZEoYo7Lr?U{}8S!a|NY%_tUMw}h+h^p0*&21q~I8U`DI_h(GFC*PKq2h7F"
        "@1Yp)!1*uTkh|0Q4WyEO$FdrW4BX`$Czv_gPFp6kXq+B~C8uPyX^EeyJA?qGdd|w_2;5ofHm!{$0YZ})tg>|;{0GRLbs2S{bpMQF"
        "gP#5H=`WEwKgBUhRbh~l!TUp7RkF7RWu8*e*q^&DwN3RpLMQpR$cr&W48rR;j%B)gBddbhKBPd?s@<1^SZ^)+#G>7>BLR)+=}s?6"
        "@kHC0m`ysHqLo<UFs)Q}e?hx`|KqPDTSjlzZpwNR5NG1!f#jr>SzP@amPQO&M;7@EsG71Ow5WOw0xt%heA)u%>01+A***ItaX6(w"
        "UtQjKn7VY1ua(eft9ldz`~stK+S8VMv=o8j*c1D7iH+%r_MSX^Fv;~ywofWn?86Zl_<o-bPHB+C<Ue(U;g2(U)*u8(OE{LYC-`c+"
        "Zbi@Eb=@b}a?){|>z?km@#CBkkpND4a}bE*ejH9ue33t{udwNGWi;WXJJpO+39@$B#)(GOt<;*k63<5oOo~k-nAurP6u=QzuWQ4E"
        "g6uF2S#t=~yH9Z`HrC*45pCogPl#ynAYH|YMH*!Z)f)nqp5K$=8LU`_Bum>)Z)xuqEusl#$yWV8Q>#x|C5aTTudLfHkZ^u&-i3KQ"
        "Wn9J_0z1WYPiptcGPNT%Elm7dZPn9xUIw)_#+=!(VFZ5jh0B!s7%>$iq?qish37QMm(NZ4^$KB;L$Q;D*&8TyPOVt8v{R}h;c2Z|"
        "pr!|$%+g}gN)MA%*U$)MXOZtl(KIG^t+Fk!I1J~DMzYthT4C7hO6l%X;OhmFDmq<DB`t-O+tNfR#%BH4=&eNEVkg_%aV+><ne!{G"
        "2=VSj-i;gD8bhtxPdD_N5&V|mya+D7)HgB4BAfh>1Qp1yFvc!7UA4hU^DQQPGSp5T*T75c`SKy{cy^oCyl(p=S-j8y<r?DHbZ+A3"
        "CH2~^RZ53ZnXZG&h+W{{XYltjj2ON6JiKw6i?T^}jm@1PR1ttg3sCH9Z9!-#<h8Av{`+&mMjV^rz@b_`yVyh%f|R8ZLg$}H>`UDG"
        ";uSpo6W@Fhy20^r;Fl)w3lsP&BK>>`SdqHqVoN3f)Js+qFN{lJ{h($*Auq)89NHL+w^@EmzF4iFuF{~0s!|LA?GexK1D(3=`>ZHI"
        "e@U2KjE~fX?6CD{z`;8ra@4>?(YC|(IUaRoP*_nRCH1frawOP?F%E?xf!-tNX<vm$(>9oCQ^%#2K=yIA#nTIX5wfo#-Ckb~=pj<h"
        "z7vnolPDZ8PsFJTY4xPGf7*>d@&I_~Q;a#(qh%xSoG1^AC~EAcNTR%qlXCOPeI#Tmc=5?l9TZx!F|AVGhTd4UC8|^+zdYq5H9He2"
        "{`LMi^3R$xcmF+gtO;|y`tl$nd)7(Co503MHM3XHIQQVN0Oj<>%?g?1B2UW`vIa{pWG-PYnAJW(-RtnMmZR!;48vSE`H2**GBGU_"
        "R1K8am^-ORDitx2UtTOyY=WdpMN_>pQ7?Q;((MuRNgMkYV8LA8?jr#`PLSTp0Jk^5?S0@J^E9!;ScI`D#e-M8%vUJDM%b#e2OY1b"
        "S=w*vfag`t8M?YxcpA8#^d#8WuQhH^soO}2lQ-F#yLxd5+x52g`?~M2+}7)p#{;3yLS!+ek(JP#BZ#N$V)7W=7@ktY6!jl6QtXQZ"
        "bhnz^0&Ryn>u*#3!1Hl%rmUrL>lxg1nyVW3%I{H%<-Z~@A4Pu$Rq@b|`M~6zTBz~yW+E2=kM6mCg_K#BQNgx4ydW+x#|0dBJN#b^"
        ")BSOnepj~RTs7_?*HWk2DIudn@z5U^`I%Tub}und*VC5=`n%Y^<kdWAp29Sk)Kk#W<iW!K15ir?1QY-O00;m803iUYS>d*S2LJ#~"
        "7XSby0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!Mga%FUKVQFq<WG--dy;xh1+%yz^uhjow%u7<HX}gG06@s*su2iX%3PB}QRgsf8"
        "GiH-G*v@nZ#DC`;UlL~~J6j~guCzN7``piWE}@jZyct?i2}62j#GaFC$E)i>ir$b`vRd?8!up!9nhl1lhq7-EMDz_;ystP>TpOu?"
        "QuZ08^!~m3_e3{HWyp@{T`S7z`|n!z_wO}Ic4S}%kBC2c3tuBb;8*SZP0bo^c(0|>nSCB;P_?2FEk{CowG@r4Si9w2$Ffe=yw%xk"
        "OJNl2HE10egztsnH>Q{`p5f*l<QbHsXzRi~E9{LmQV=70@L|^RPO3vOC|OP{oMlCMFkGJ{9wdb55XdUoH)1;~UbyE$I{wzYalUDh"
        "%|gB8{yhRedf?T@eXMx}2P(4OA3$i24)Qm$=l=A;M7|dl&uKN*jD~0CF!0=lDk~e%#5{TGkBsOK`4ww*;IrebR#JAwR+Ra(?6H5W"
        "Ij{34U&FWbxqAYMbv3pvadCj0`(qS}lNZ1KaQSFZylKUDXTBl5B(?n5qxqUhMP9u9l|YnhaBIWbwq(_{%h7e)Np3SSIXy{l^GYB&"
        "JxPgZ)(C_KH4Q;;60n8aPmVnh;L)l0OlX_xNs3naFosZOUWf(hiXte{N&`}4e|lqP@9hk3#~;W*Q1S|n-$-*M$G(24lvD}*UYcEq"
        "L``hA5rh)aa#U4ZMl|V04T5ikHW^J9#-3Pk-eo6>y<_djgGNmAR)D2e9FizraYQWZ1K&ukSd};v$+zV3Svha`))j(4b!-$Sdc*;q"
        "Iy^YMB<^3DAyUX^IdS+2@o`u25ypk>nP^#w!HGDo>t!LbQoVhUw8e`>$w!WW(F8yNaWlN0Eb#i)LANE4*L5tLCr=AK4p^Ib{i%wg"
        "b{LOQK_fBHQzkVhR^<CDkcnDP`jx<CP!4&g1vs&_%@rE_vzba%#|@MyU!Ck<&+xW3kW7*?56I10xg=BhDX=l-zo?PV$q;9E<Q1kh"
        "S8GU&Q+>XEtBOlM&kB3So8|ebxT6dDS`5WUscM~D#f{)aAPT12<0HBi7eqI_H(biORtK?*J747Lfhh)|;!01&2*Ns>b`EBZ;(S92"
        "lE>)@UANeBQ^3F?1Mn7^j#Pm7g?e8Q^zNc2JRlbj9(?)u&fH!e;}3n+>#=EI!h$q+;@<sy`6Fo%ORkOtI1cTor=ntk>H%tpt0n7@"
        "zSh)^^&>QUY@7-H@rM=@{|+eb-x^RlHVjxza`G4T*+8G)o|{(8Iq0N+|Hm-#3cjbFfPTPK_%!S|ZU0By#1YC0q_8&<|6Fl1s@@%M"
        "$X_!Z<u?OYqJvxl?^|&_AZBxPqk`&LxJInQJtK=+R3=H|%id{l$A5|seO?x}RM&9l2-?(1C`1kF2y-a*pjspB)p*WZy$b>4aF4#W"
        "lLO8l8Wi2Jfj_$D2eN0asBPxMOvhckVRi<zjDt3cvksiJ7aA`Fo*9h=VqRm0VtPl9-@blXfEcx|<*0d4F$ne1;onEowQfw|HzF9G"
        "pps1+QR0IZtsb>dDS7xELHExS?K@|opOp02ZD306M0<FI;c~+eIP~CaB`G<`!P~6gA`l(A=ZZOxwopE%$@EB#fMs<Jy*IZzS!U<u"
        "B(YSF2>tox<qyAQctX<*99>)NxXBXa=5#d8!)G+qXIIBM*lBu_(BP9STHK)TLZ0=afmPA-jZlueAZX&HBd7VS7F(c!Q@=}Q5V(u4"
        "o+e1+RC5g^O|u=pakrCYeRrm2&;t_Wo#IgAF5Wq&Tsm9gkt&3WbV7H`K<TBEOPztYIQ7W>Pw=bN-4mhq6dVmAskMN{YodixDdn}z"
        "%KZgBJH5V|R(C&vO><Xd=l)_ctdbqLTa=;&gn@NIsrPhoLO)DuFP=NyqpD@~cXe+nUuP8#VVHT$6~V^9a|;UlT{&OVBP`T@agOHp"
        "uwm=F_ly^_pCB)>P4n;yCGhv`CY?SfKjCW=iw?qY3eY=J!oC7s9zjkt)IUE30|O(VHvH-K7TlWbI~eDNY|@zZhlE0&DOOxkM4v@r"
        "fFn*q!gmns<GNT3S7$AJ4Ve+W;|ZW9P-|o5j~pMyn0$UYlR&ASFtA!1(s@R&hH+vgpeQVsCO5&yYTvO6tTp&(>RHSunR}y>?EzC|"
        "q0#kuPM%ymJ6a%4`nLnf+S{0UsNMKv2rm(Qd&WV^<~}jw0G=8$+x*fff&)fy9*$3=XcIRJ04}zQ)zMVt*0CVZ;cih;Y_{TZSs-hO"
        "1@3nmPJARS34Wh=d=gM64Xqjav{w4$b#{AM@A2-3-~*J4pCV_t@1ma^m<hwXJ5%{oDaU9{DXb_kaVJnq+r(v`&<)~|CoW&p49`;+"
        "B;S`oC2vEvN5OZh^ac|KnFe4@^z`<f=aww<!^R@R<_saZojH<QtH24`dTFz0vBb|YIJMZv;Ha0*GJIGoIUEY_Ll{JplM8+|yJIZ&"
        "V>-C`Xi$zPWZx&BN1e()`Vdcg+g6siGZ{WC?#9m-1joJOx%c`!A^rnUO9KQH000080000X02N1gdYc6R03;9q03-ka0CRF<FJftC"
        "Z)0I>XJu_|VJ~fOWMyn~FLGsOWpZV1V`VOId6idNZ`(E$eh-lUK<G)@6g5t=HZ$M>>);Lrns!Kny%d8$OSH{JA~ljq;uYI(-#Mfx"
        "QHq*$1`J3%7oN+<!%tBZU7emUiQ(cv6cwecPaYWG?G2$?vwBk<$W|(1m<E*{X*esGPKl79Kz?O)P05B9cN-~K7DdtV(eV+l8>tL2"
        "QWg90TP6g!B|@OuR>_)d_*NFQ+M%PYkYdYsq_g<#`qR7jAJ#W1SwA*R@tO&9)3D+zmPXOSXm5NaOICeqP1Bl`+_1mS;QyAIVxN**"
        "dB+6*o0ZqBp<JyWY28$8&TB~xx~?h;(R<I%>X&YH&aB40bq2*_WeG*=gQ87S9r6dM%DflmjjVWac*al7)Q*)oH3ka6r5FO)GQ~u}"
        "uB^h(UOwjvio*r$mdVYZtT6nZ^+7dQmAp_=>l_>?*&MJt$45m42q3orjA18?C>A1vhqhwL^7x3ruVh=2=dgjv^H{TLn-UF;T2305"
        "l6#0~lWQpcl5k<jiY(49y%+rH77XGf>%Ap|jBl;i3<7~LpdXri1k?rA_=7|>BfrzCWoxCRiX+Tm05j}>X=`Hk43OO_()l|nne%&3"
        "6{o`JEDAQqtTNmlV%8RfUh=?8v)8cvC0d`ojh5v6EG5zU0v|6f{}~j!ee#mmpzX{)hZ$v9m_G)CE}$8MPBJyHLHcbFW*8Q9Ywf`n"
        "4-H7ecv3)tLnALASEtjweIPHo3?hEPWmV^UMoZ`H0~d^{cw`;A-EgW`?`Wmj)Xnr5fxl-CSv-cy({iOV3Zvxb6uML<BEwnUSJ=5&"
        "+1||vp7>6lqNbt;z@{zNEzD^RLm39c+DSj4aFX4qy%*3MUgulJN-%t&s$5-=@5VYM9DUiE)oFGLCAU*y%at~(Th+4J*unySEwp%)"
        "KuQ(}Ji4h^F5pUmxdu~9OEur#C!=rm)v5PP_G3wivav+8u9JFoIU~8d?}YwdnmLM@SDM)48g4jpJWW>G+bS1OJy*=@tDBGT5}u0$"
        "2eZp!OS(>W6W4ZZIlAi-a`IDn6E3H)!t)n4@x=oY+g<uKE-+$V>STH_RFU7YL+)olP4(UC-?4GuRV$q@j89z~3Jk<YJ32O9NPcYy"
        "Y4Boq3bLKTDxOD(+ccd6*7UT0V`hV@-{V_~P{|!Pee&J7iKm!}C$;*%Pdf>GDP~$FLQurceL6=3dYKElq8lJ2N#ghcLQu6LSN=ri"
        "x&~?$&tLr|yXOU5``X~Bh>E6-EIdnIcw7Q~{3ICB2LX4p;e`kID<qiskisQs_P{+!19>2&fu3%8g@hEj8UdE33#piPR1}>$-dR)8"
        "(kj<5|Ki1)t{#?OK`=8E2Xja=BV!kKXTTg)#8!MBN;=Oc<EYQ$WD%t#23D}Z|2h0Gl5XMjhm^d*m&GN#{fH~ClkhOT?EK6-2Um;N"
        "q2o*YayoX}(FNO3J5ED38z-^P@oDxBYhYvR=8jz_LUsjI(rIKBAPCtRej^}53(c)C^9}X%uAz$7%rJ$et++O^-6I>;V-Mn;q9q4Z"
        "*(UVz443VPO1}gIGxQ7WaiSeW*cAnIaAhN96+7H;y+tC*;vo7YL7dqSpsET+$DEmuHpx&IEHOmCh!OVbI6om5c1-rVh4D~(F5APD"
        "6taeZsJ#^%1$4ZCvmo#T3O6Pg=&Rm&Pi4E~|7F-EV*Nn{!fLNH0aXeh7-26!HehZz@=>Vg&;pw7Ar!eelIuNfP{FpiFDZ&LD+-Yx"
        "Y=_B!Rfr(Sv5rUTW5#FqQmh0s2c^+9mUW-F00Ac*-6)-OoLWa>-0+<`@0S0!TRX!n85)p^U1;lg04tCWIBv;v=pGefn2ovHQ9}V~"
        "X~VTFD^%V<UG_CqcR71xMQfgD!3WHLP)h>@6aWAK2mk;8ApnCIu8%zi006rX001HY0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!ml"
        "b#!TNXD)Dgl~>zt+eQ$5=T{8Whe|Ci-x4EG;v#XIv_&or?9?xTK`+UXv=GT<cbB%6qW|8RC6_nJNo&Bi#GRe%nKQ%5Wb(5rEIwE$"
        "g|^a4RmfrmqDbi%WR_LNYGLIaLW+q>QG>y3dGPic&{&xZt28X>_d+gJBC=mG&&50lKCRIpR90rB6Ah#a3S}Wvf)s<3f><P*SQQ9#"
        "UFKzmITn^qHI~XqQoMykBx{@pS8P&POXuXEw@hmBA<3!~0gJ{r%a21yv@)i<5xRhK63O5m<!WusJeW)-L9o;+hg4XRWWpHINRB9N"
        "yGONz-Ikn|=I|Gp*bx3GS+&SQ_*9YeLC~C%<H<S*f|s2R_ZKczWih>^=q`j!){6k>hrFOCnzB_`GYI}5!#nafW?3K!DiEeF*TQV`"
        "9Icj#5B$Iv2Olu0*sr+@k1aGJku1u}Mlv;v+Qs=NER50vnKollF3e_+U}NX2%Dl`sm9<uljVQxN%6w8k_l0lZ96k_Gb}pq@+VwYY"
        "1wQ|j+6L{e6uPazM~{62rYg%!QlhQeU;);tGA`|<R#Y20`VUEe$w-t5RcWo#CM!#VbCy&YK{WC?;%16Acaafy1TZsoqWnX~^3p~b"
        "77LKn7Uyt$cKYl|*b1z$=$}44KIzViDvvU;MQ!}?>FAV9u2H1z`LphrpGT?8{n5$Ov+gk7Qh!Ayc}{Q?olGq;gM*)3cYG$+2w_R>"
        "Zf_Rt<q5M*&Nv>DN!Qs#E2_x0)UU==Ik#L@5V7bSmQ|MZ1ncPLL96?9%><^Ug0)OjD&QlG0B;YkL(c_dE+L8&4gp=Pjs&xeXpKou"
        "-J#~x%xm`g-oSbrYe^!lPQE}9n6=Q3L@7=py;h&&A0!1tjtmD;W!&;4)#usJ5vAwIL={W9s<iX17w#*)GF@WFfbpDe@UM#Ufk01J"
        "b`4XUujT}#4O8yv$)-=W4)10*A1G}>_O4c-mi!3G9cly4Go*64ghQGh%u=;}g5rK{eOP{|@|dbb3MuZwaE<A!hlMZi_;v@|?*0b1"
        "2X`W?&>?qpsRQ$GH6G|QImXp086;fL!G#@wU+@Db%rmbUQqz-q=)E~UJACu;_-xiI%_Z7O7o+N2HAB<LFFa%lcrPztO0FBIXi!>0"
        "gsGNG`2e@KQ6Ask&gz-v4(huDeKZQH#5Q3h<dRq+151)};}B$(6&YM#zIqQtE)I703Xm|e9cc0xF{G0nhcBpA<0pIIzP9G({bp`I"
        "%ViIK@B)q)Uk!Hn>o6BuNDBFyOZHl8rKgjtAx;hC)D94H`xFJPTwly5v%WndhiQkWaDa}mEqgew=L}w)f774Asf+a;TOq(#1r2^G"
        "W6w;7sJdbvBAmfC4%cNah=J}|u!mS(>$p!YHfC4&VGtjE^#RhyQG}h!@8Hk<<QWTslq_)3RL%L_^>y954AKU9UD5s6Vv>%WVRCQU"
        "tNo9@?6bz_x;_IpM*;3vp2wtlkGVsb9v_`NA%*;J){pk1pU;LSxLxc<#A-OK(I)$xS=R9-wGL(m@7JiY3p%b>+Z7Ws5AHT_kMRE;"
        ">5fJh?IU&`jo7vIo<z};gMB92i)}YYWx~^Y+G;u+YnnN;NAn}@xv>?_70_AG)9fhZ?*YmugL{7C$>YgltT7zk8^<u%nB4t6QI2%<"
        "7AzMcUGg6vk?dusrB=E%NLqLGpRewF8hP1^Whyl$^^O|x%yk=a$1}vi?JgJg`r%F{==bLOTW`<m@4AB>>%Gt$UR=6ec4y234NRHR"
        "0wigu)4oz7fh_cX^rx90DS}3Pn$lWx=&nk;1&+aQE2WV7o@umW89tuF@3i6+BFA1-Cebn`VDoIFfU!usotmfNbCC1`rm&u4sWc+a"
        "kap+F&_H9Y<VCBF2905VXy3Y*MXK(ZxQwLDWc-@Be{?m7t&>SoX4+3g2r0jqoY%*(QuMZ07vjABH~U)o+gIwSA()7WHnKvm(00XV"
        "a^8KD>R{m5j07Hf@NfWdubqQeo^YUIxOrc(8sn^Wdm$-QZR$M>@-*-i*>EV=v-W%5PhR_KnKB*vKt3-*ShQZ>?C|@pbQj{yxG+5v"
        "VP8Y=FHlPZ1QY-O00;m803iUz0#RGu6aWB7RR91a0001Ua$_%IX=ZO@VQgn*ZEayMa%FIDb7^#Ma(Q2KbaiBTE^v9(JZp2~Hj>{{"
        "mH)uwCo+-b$7GU;_q27J#IrYbmzk;YWb3LN7YiamiDQaX2-3DP<^O)&4S)nekdmG3ZJkmRizK?yX!ILR`M&=yP4leaUd4B%$m+7%"
        "d!lZVy|;rqtYURm7Bg?n3ogHVtVp~9zTR=~$M?T@A}*_}SbIPI_nU?9`(J+X<rmp@S5~#hsx=@GyuGnuVv}d9_WF}3i}s=v?N_mf"
        "?`c(Ty|~PC9^<saQxAVEo1*5G8n9zEqSO!H!zEHc&sJGl#w=g+?UpU#vPiSF9(Vaa?=L^R`T6qg$LQ_fe!jfI1=bON*zqdc@}eHW"
        "P**IjTS`6vf3oWGfo*p=pW&rq;One^eO0iX*pyDJ39BV){!nYgF)PX<gCUVnkP)(O6x`T4g(3Lz3lBb5?|%94<7M>W@*i)m-n@Hz"
        "HIujhp4FP9D}dtaw%l+Yb?D$L>~5l-SDYo0D4QyVG3C8zZ@0s1CT=4n5}(84aFpb_yyZpqyObzJbY24fJJ^9V<4L4O;7;Rz2~WM2"
        "8#Dz<Pk{S>LE9dB^S?THej$m(NxdaW+ydtj1i7FkA?pKP#AO0<y!4wooxk$KP_M3Ts=|R-u~;L&_K^?%%}iabnBY-dZns&DH|qvu"
        "B<PDs@YPo-{0Y6ZtUUbYWrc#qE8*WL8p4hR_f=MNd+Bn{P%V`fh8tPbi|uWaRe`(^%a2t9a>*YuQAg!1U4{mX#7*2FJ|)d|Cjvwb"
        "K!s>PG?<99Z26ky0uW_I0%(`#q4%}t|62Gs%3`96@VrO4lrZsU65nl%?=L=;Sz#^*1=Fz_y$0F~7=v4(NgJ+fuq{d}l5EXIZ3?AY"
        ")_qoQB%S4s7lB``{Lo_pEMk%5{Gy{+iJP*N3zP$mRgRr`zSywym*4u~V#6QQGlP_h|JG!%xPsqDWmE4O7d=@jsUdcXAS9#CioMhd"
        ";ADlY4?~kOmI>~m++GsUWmT0`;J>3OJQxmk8Z8jZQK|MG+BBAd-DY0Z6o2vJO<4j0<^f@zBy3>{NXqMY(`h|yj^IF=*ztG<7BSv%"
        "Eizw$I^i(2bBadrJ@0n)9QJL?Ylc#L(jt?{J&?Wq4CV(L<88}?=iWWf)*GO{yf4ryNC$@4$ZTzz^%{0PWw5oIrgbu#6t5x;gbP5B"
        "Ei1AVHluH5ItB(veGo;or%|Y++&l4nxzvu_NTXUHBG~8&Gu4VBQH{dOQx1q@vL@HazC{9pobc)fWIT{}2o2r=K`?K4960J--<TVZ"
        "UNNqD0_)OrUAL1gh>ua0+!*Gtm4&6;I9y}J?tS<>1VrzZ1BV0u1T2UjMS4v=R7V>F&4z+)VT4o<<b+;DYg<vuw;=k)O7SB#87QaS"
        ">OaGewfVttK{GXX#qTl!?rqr((YLe!)BnT%B-W*JT#_s)2&;X~MPQACjnqa%+6OY+JsaWu+fp&)Vz@4;-v3~E!<A-#MB+S>O+xK$"
        "GqGi$M+@JYJLq=&gcctvb1W$(LOMln!;Gc~AVh3UKguDjv>%8CcBFQFs=Xqt7{M2o;Sk2+CT$7<ch5*t6>1!Zaj2;oned9T_TH8S"
        "mxEf+nHK@wBhgD>BAJ|2MW{XNLBlB=kL2ScA{J>|0S^HyOIvYK=65`3qnd$}kbl&3fngz_up>oD&^oIn<*=SSFYf_OeP33$OM2S7"
        "rT3};++yZ!vIJrV)QwLZBk3jBnwk2ioH=j>BPn>KB7IG>;2@5m3A_#rbB96510ZDNvU$;LF;C%v9jr>q@mtTUfe;Y}@2`qqfGEl{"
        "h}t|ro`bKJq=)SQdT1%o3TC_QZq?P^8i0W@&^{I_t9>SFX)QzL#_@}s73+qrIcl=M+t(ZL4oB%S<P(t`PZQe;jNyRd`!!93-)`C^"
        ">t~T`?d}O%5P{hdNU4N%Oic4ZoKQ2ckYGM>8*!_TnkIp;8WJTiH3kO9B$B^Ml7#~`It8>`ut^T1j~~mr1}&d3a{Lb;{!e#|xIa{C"
        "4{=mKRCA#RB+}atA3GrQQcxd;Nl|~bV3YOgEKKO^9lS7+qtdsiXw4Q5il=u~b_bgP0HZmigR6*bHa@-MQD<_8&sKRE--6|(nU`oK"
        "q{W0(%zOe#l>?aN8SI-_Y7i1#braIJA;6YBL@H~Swq8i%r|*%pTX42#m$AumEZq<El?B%h!dM-^mA!`;tKD*pv;u{{6)X#lkx7p4"
        "0nOo9{%5xk%N+Qk1t<M*fYVqrLpDH++hdS2g1hjpKl?}@d`lNZsOJ;^i<)k!?~gLSCcC01Aq=j4eVpDikoIf8(>TE5-@q1hoIgQB"
        "U~}FwN90fzYzLWOg88V!7%P1;Vh)F%s!=Y2Ty#2OS@Oo35{1r|HjX2bB(UwYHO}QfzFvA~W*!mqUF`*({hoR6uGE*05bc2=(-mdL"
        "9KTtMR#Af{_z7YO0SUQzkP8_wj0`lr$OSs$1%#tP@Ak~N4X(yzP2`-M;ogH}*Hu8P39m0+o}N0T&M9<NnWIUyi+I;YuyAvIuJ+Ze"
        "ue25+@N4jy1xaD<qhv{(AhGnom-(L0@4)ydxs+qWnUd2k*p|{_Y>9J9iA%|nPgV&6V2xHsfNdB`3H8$4IA5*|7&p0agNr^c3bzaX"
        "n73Js0>^PDjX=&Ev8&3ML#%HN0nNoLC2l<k=|e6+#p62AGT$2k%2s1!kHKp@QirnDwU&#Ug@Dingy;g8auz~bBeQo|c#;!aGk;Q*"
        "a9pY!VLtD99vCv|j{M&mR<WYa3a%21NL7_C>_BHE1yM<8FS6;83vsjxzgx`xDp?nhn{XgKSwdxD0PsQW>MTYt1bG~&eZG&dh~wJ}"
        "XWOhGpa>xvWRdjY#yShMgiW49YQn0hiyT>#0K-g-t>F}LxkJhFfBE?O6=hvCeI$=RAWh|QL$huf#yK*l0`}w)BuDWrC`H5y;1v)D"
        "fso8T^-Pbp-Wob&9KA$RN2DW}DUH#E&nhc=wOaYqS>JRd;G1U43RKrLdl+QnBX#>yo)$$SZBHvWunz-#8Rtzx4-YE3iugzH8@iPC"
        ")Kb8|Bwu&r&_>4XOnlpX*C^L9RcjtiR}G81;;GEb0qPp9*LGL?Sk|NKRkL320ZC3WX*ztCg7FyxGr~^?sEot_)-Vfafh6Qmeq$sO"
        "ljkR-bSpr%*|O^P#FWdq%5rClkeO#6qWGHX#SZL(Vo^9V$S}U5Bd;1VqZqcm&K|sa!#(00ie%1F(&VXBU3vn(i4>kd7L_)IghK%V"
        "p^0hK!U*p0G!U@%^}$-EBSP`ZJc(4SiiS(Bz!4?*C~3w(+Ev#$L`X3oT2PZaBRG#8E1DI$9SWjaK3W6q09wlQ(l<BNL~d-3Dw6a~"
        "g2g|QSgM^s@MVWm{wiCeKz0(eWAUxM6#skn&DnR~o~PfvV6WoU`HR)+<*T!^)pswFmyE5>zWs)$ar)xBR|7!AXy2Slb(~quIO|Y;"
        "aB&sdJ#u-&75@C@<A5wOwtDZhVg2e?JMZ|~2wZ6lmfP*DP@aN~0)o#6ugL3W_axwJ0vxOcZL__Y5!u)^D{$A(sWecPp(`qwZeUn4"
        "fUx(d%-%_(@<NW}tsC>4q+hPxmF7)kfJdjKqjdnonL$i4LWH$_F7tL|O~9%Lj8*^~*}vrij3R)vrH?k5k8UCBXXbLn#^b4U+02Fd"
        "jB_McR&I3&R?T!Wkw`f=(ZV$vRCW;7_I0UP&wS65G8V04wm=RhT}md5WafHc<y0U<C#6rx9tarXo4Sn49LZLJdXlB7iqRsLFrcsc"
        "3!X~X5Ln{Dw1CiTwFjNdb9_QOFE%{!-qRvqEWTfyy`WAf4ptg?y(iU<a|Q?!lLuO|#v=63Y8wU&I*Qqkt}s5y*t#eM=1F!CuYAG!"
        "_iU#_>ygO3L(;Z4Y5Z|j3W08wjt<1EXL)~P@~%dIR8ai?BD`-;4b*|~Qd+X#n0ewdY%A&uBXXXO-?zXtVMO))BzjN)?8b>E1u3BE"
        "7wg|G=StNZ3+GgufFYSAos&*gZ>_Z?*$-rJ`*{x2I(xu17yAGdbBpOvw<bKc0M_2)IUprWdPv}x6^p?K&2VGX2|INVz*Xdj_QC$}"
        "@aThvnuCC40$Us^URN1vo|Z2<%!CIx^Oj+MR;MVvbMSFqTKZ$L6AVo~d06=;@cN0)spt5f=bXlz<3GncnWv}I?no`||I!_mpucfO"
        "oNE1&Y@o9SZc+APwExmYC14y`Tt_52)Oay)woj`0J=ryBW=*W2=a9%h9#Kd_xoKqfto*ZdZ`$!lJnFcb@*s%c8-T1p33TlZ%xRO?"
        "o!=?Tw&g)D!)|(LYcOt|BNS6&9k!g~Sk$Ga<S}HL5IryqTeEJ0t9Gl_<B-Y0L8=VOqctS+vOCX)1q)>z3=K`(Rf9Ft9>rkD1jnXo"
        "k5G5@*4CV52{mTW8IpPOabq?|8YnyRQED)a0t_Lw8&QOlh+!Y0N#swu6l)@Fsafr@DWG>mFr*BN|0+e5iSw+xA^WoQZAe<fk#+Ry"
        "vPRe2=V?wzv3hKofD9m_LvP-qnPk9wV|!HFq}LR;MR{MyqR5+{I)1b#F(ei8D}+iaKgY@lF4#9&etgU)RnK^!^-Ik>l+A-)_NNw^"
        "6u1mKLaE!m_1ZHj_t>O|1$DN{>2(B_RXrEmo)!(L40K<Ssc}QXRWFoodUcXMXmEunp%M6kLtT0$1eD~=j4(iv)uYsKc*ew5_1<lS"
        "Z!f$%lCIkse4)xPp;z6CxRzUL|4{}|<^&y!uzgsE;xvG2&RxBk<f)Zfg9WOaK$}4vH^}xw+oT$1OLh(mFwD8M&~@3G)UGk%a8#!2"
        "!;&B_VzvV<)HdDd2sIf#-4BD99O5t~$DJSnO34cxrOR^NU`G#450)$eDW}BJGb2_FP3@Tbe0%5>M_Ov^IfkPqI&3=HUF|n`v~Sb_"
        "<0{+Leuyr`cNL5kv7J^g@hn_AGW7`dz;%MI&1P2|lX2J55=#}0xfUtg5Gi<S&*{tt#r_l5YGq48Bp`h&5*28xn^PpKrGcrS?%$4S"
        "dd<aMR+R<jyuQ8mVV>3*y^#ngP=CANM90zn#I#k$u}FKe&Adm=H3Xg;EFj-lNB33$sY+C{!3aJ1C*qh6H@ipIW<-87fq=@`7YDP4"
        "$3FGDrrMQ4)=AB`MM%R*&#I`4i|F831`+hkcN-AZO3^7$q~46{-WI_>hkenL{7@H5+kWUMALQ^qe1CyKVZ)^t;`^Dpbo(N{A9^F5"
        "5U7_1kI4zLLiflUZyM_Il}1BWZwv$N5Jkw;+5)bk|1KPUs3?>pj8-Conpq=6+Lj4L1YC$(>bewaayQW;n)mVbpGa160)Jt@-Eq7O"
        "bb%tEud|fiM$|9d5Sn)T-8D8Ms=HQWB@;jxK&}dO9J+osWR<~8ql;d>U?RI7b|uJ^dY}qz#sjA^C}tbO%)q223k*N(OgD%W_`r@?"
        "PEsv(WY&T8?WcNX9Hslt%cJ_v+wjF%a+&T(KceBB8oTbNQh-08bw+&|eS?`1;0N!`uu*tN;^qc&wV0y#3~$>C*=mH&!PMEQ)92G5"
        "RsV1Y1&(^}NOvz*u+0)%|K13YmiJ>@f;C)S)*uvGjZA2L?}&zCGkbn=i=@^0ILLU%+8m{0#40V;#IU$dZ$FKUrj5CfGBYv+B`GqR"
        "C(GR^vR2X+!QGAHn=JJQcPT3x<EWO)7rSLgLieJ#-H>pIQ?eCTmMRC8MLHI@ZaP|J2#0E4pJ^%AuNDV`9@I%)8RQ$L+Pd-zhoUOh"
        "Sa%Pr2?@*6rQ()OE{{-DrB};IrGJi6K2ITgS^WV;bRCmh3P@XH_FKD-!h6c6K)QP_2${dOgJpHKKkjj`raw7mbyfLBju~7PokOPy"
        ")6v$%c>-|`c+&+Owz_IPF;bS6xIr%xi$Q2`q|ezJ1ZG`vp6nr;0FQ_+7P~}ItVfQIbr6lGPRNqGC02X8_v9-<Eq(e&HWq)Y)(ysV"
        "@99a9aG9QC3K~U88Ap+h44W06PSG&z><3n@<ds0EdP%m071Zi3k8>0A4O`j9N*U>j;3aV2)7zGF)#ch7Ov9KWF!1N|R<q}d1zx@1"
        "@umJrfZ?PnfO5>Lp7{NNgBg_tp5bwS7IU$bzsO+sn1_?o53pi{cQo-sn%c6Mf#LwnI;&kg14&_^&~z2}QW3-nzNl%`-VUIC?xqY7"
        "NkQn|$OTM{v}Ebi3uO^Frh^+sRhqeajr3U<sA+vJo7SheB}a%8qmzt4mBlzPi{$n5nWa3@T%O@6G?YPUD!)1X{@Vjc#(`g)9};19"
        "KLLJrdU|>Q0VI=iWZeXoR|<tuH6K;vc{j)(fE`xrKzC_iADBLQr^d|@)t*3poOVwjc(O)K;5t^$CQv*{)Y>MbiLL52_1bXHWal{i"
        "lesyL>R4uuV|WrTW#`zCkFuU<4jO8gE~y@*brfEQC=M?(0z-xBxomeHwVPs(r6gV24<|87V~(mgZ%gizHr<Uz7hLC9aD4{6s^05G"
        "rFqFrQ+YZGe5}mqAknek`jKx9{R6u`ilVFX8P;+5yJt_rsdAsWQG)ve1Qt`s4neEp-7}$|Y)VEkJ;@FnWbA|sJ^TLr5Ee*WAz-?f"
        "s}oK%FV7BQnOMmo&|_5uv$ZY<p+AU+ilujIO#eqTbPB3c6Y7WT<r{Xb)BVdY!;IJ}ko!N#`sm-WK03cV{{jCEj_LFojwmI{ANnb="
        "6i&x}vPk=%ucMFe{_FCsJUn%^gN6T;^?dUK{`(iNhv%-{)Onn>kuY?dsUs{|a<P7=W#;;gl9{6wF7*E0J3H!EMcx}{sH-MfW4gO-"
        "MD4A(M=3&f#|J-xH27JgBNyBcqK2&m6R&C>JA)BXkKlLpkRQS>gBSbfv-&-NCXPX$#PE1fKA-rZ0OwBwC^NzzvN|}Qd8yx3P}NVF"
        "rYy^Oa^Za@(8pifAHz^Tw|6SbEwIXnDrZqdD~uv6zC@8;q?DAay{P&25@(~@3`Rcx3s6e~1QY-O00;m803iUi`j2nk1polq5C8xi"
        "0001Ua$_%IX=ZO@VQgn*ZEayMb7Ns<W^Zg{E^v93S8Z?GHW2<EApe2Tz9fKOx^8QS76FztZjm-gv)SD+41+<FCz}f;Y9wXHE%M)Y"
        "M@p9B*oo!>jYv^^?)AC5BuTD$29gPrJ8w(jmC4ym8*%Fx&n1JqLMx#>Q@Mw^<?+2KbIGj&H%OA?@Zj(uwI*YddoDD04jfZiVXSYq"
        "=+NQ8FPrrde(@FZlZh?i`0ya?v9nTXc}C!G4i6aqlzhWbthjKV`_i>HOlbwQHj2^~#??FY@9fLP<@xmN^7Fe-V|K!h;<*N=+Mr+Z"
        "?#1^>k0mI%h73?$NlP?Ls^)`YO^dO*>t{xSX08CCQYE56$Wmef;<hkoCstUqgq-IBl6e1d?<wH(eZbyw;a!FO8=!8=8@ukIS2!K+"
        "u^IPb5#S38QbqS|zTA(Fm@ju=l6!{2Xw4C)!o9KJ%`N2WD~5>&j(XaDYOGNCJmP~(o>TCLePGqq!HhMa%5#L8aR<YT39G?v3`j)I"
        "xc3O&(Y8O*orUHC7Dh|38)z<h;qlrEc4E28Nhf;gz_a>&fJKK|%?rj$seDJ|pORi{{Xm4)ew4-eAVzC{VgdT-a1?>1fT3?Ot+-Zm"
        "GLARQST$EU*U>=Cq$vvR>MNd;?bSYqxD~ie`9)=wOCu0s{Ku#tc3D9V?@MLjVXQq7YF95_(@i=0cnMSVT(~%-ksHR{3hX*8m_>t|"
        "tiJ*^UwD>*oukGr{&^<at|*;aYYQn@pk!?CR`~JuDMl2Up^m}Uv5z64ok@Mh?|SURucsj#xLao#c&h{@`#;ebPuL!X+LdpW5RlD4"
        "N{qca7RQ_0SH@y)bBiQRf#4QMyBPYrAwrh5)ZGAV_Uc$iCs4#_l>;Nf&SrWY6q|8zyW&<pYvWyMRiR-Td4-YsAgj-+bgsS1i-M93"
        "t!vl^$TH3}Ut_j)%B?HPt$py}Plfuv)VjJJ)yEgM@;H+amJi~p8}zIF(1O@;!<$A{_0)k*hr${xai4w<X0{tLXW^Sp(372Y!dJXC"
        "G_w$gOQ@shCgiHg(Y3BDz%G@eN+0CF&#l2@-Ed#-P<ewJ9-t`~CRG}7Zbt7X%oJW_MBYK``V*#&DLO*`@Dy|r!0@R8KNY@2y2>CU"
        "%Mv73GIabJx41(FC9`!4u}O?@Qota~PWOo*mGg=UW)te4t8W6ht!pMKO`TCE@QY3qrt)fcI0tk9-^E`Oj?wP>DHanp$N%J-Maqo{"
        "fEv6-0jXI-c|$0&0&?2#T1j3%l&Kw@+f}ZK`fV}C*xM18hSJ}UjU972FWka-YC^G@jD`5$63Gy$I)|s|_WZ@q``a5_-uapdTlUd#"
        "r>NRhC)i419a=rt+>9thpY;_ixh{i;s**P?)_(EWUl==mmDfJH31^rS#)5r|tbx+a@a{#%?Jb*QG%~zb$=HSdFUL)`+LL{b+XQ;!"
        "ys1kcbt_k)5tMyl7S#8*_-%|dpmL#08Jep|yOEWs#iJ6`v1>Wq%8K2&**d^6FZ__m=22Q0qp7Sqk}1?Z8fijVSb{c4mi_Q?WU5Zm"
        "jUqemBaBu6U=2r|o4y{MdXASNMWETZvaC7Vi7iIN&mCOZ5}}zE+@n;)UN6L0d!+a{{(OCjb<8$({i*aDM_X)}pqY?bjI9-SeaBO1"
        "4^WOT0sY2f8-nv;oZzNse@-vXagO^ICe#6qhvA8)g6avAr#<xD$2l<0m+xa6eQi7%U9Lyg8heDRi^gDK@3<O2akxFrLx+$@r}mgM"
        "qKqf&DHzAG965@!l+P>;g!sypLZFA!QZrlT^^i<gN=VssN*&U4dPIFeFUmcwMqNYz&y@}f)HF$IVdxiCJYj8f{8fMdcXN&|9@P7I"
        "zJA;~Q7aSa<AkLN?zmyJ--j<3=c7R;zx!5yJ<w(a_UQONY28DO#nJI#qf_&WerdA#`u<JrwrWhjXUYZB)h!sv0s9u!zh}wzMb|{h"
        "&O_nwqQ~O5g~EXO4^T@31QY-O00;m803iU|PzfTA0002J0000b0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FJE72ZfSI1UoLQY"
        "ZOt(b!Y~Yl;k{DtV0ot(K-~}*fDNI@b(3O9;>t;pxIF=hjqMGe{)dPMG0$~(I-Wst^%jVI*+bZ3glZ&N@&e{lTLlPa<t_OJ=PzH3"
        "s%VBODnAB9?7O}TrU@nq$`lu$5)Hr5pwqp*$<r5*w5nBDsq)RSX(_ICP1o0_;T*;29E@<mKLoFR_Xbc)0|XQR000O8001EXg{`{f"
        "9svLV<pBTyDF6Tfb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD?)Fb98BJY+-b1Z*DGdd2NxgP6IIvhWC^19jr{P1Xl*8Lt=uC4A32="
        "Ip-Qk;=^$y`u6yG1r@28Y{&8c`HwMXcX!_dO%f>E2+;~M;P-?a5@vaTOadWI#DW751LIc>tnvN`g9c9EZM*A?F>TW}OD71iJIuLf"
        "O0W-}oQ)%<WV_<5_Y{ZoP}riC%J@R^igq|YGZW7*+h#zo3PnOxD#d$`RLZJI9TZHHF-Ykbscwh45Z{2HbqD0Dw`;h4{8#AP2J|TU"
        "w$2Rwym$;Q2-q0LJR7h{wtSY2U{Xs?q+s*#s{@YGR7GpfQOy$axHvJ-Wg7}!-EXxtoW#|Ca|7lHeaJcoKB{i)gBpCf`^A^Q9z7|B"
        "`WJC7sOTS0$Q~K9!|n{RZkr!aO9KQH000080000X0NyfRYhwxk03IX&04V?f0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rTWo>VA"
        "d0%jHZ(?OGaCx;FU31*F@!d224=DXmQaLlSo9Tm+%Z!rNPjTY0)#SmW;owNzi4~8a0Z{hY<Nw~>1xSD(?<A!yUwkBX7yJDsCrR>d"
        "&zpwH&eZ$*GdHztzy1B~<!ja{xe+VYO2f>Jz}KoT@tD1Pb;XS0bt_c<^vTmFuVp3l1yizbs{*j<mPy6rolp($ik;%!jn3H>5NWtE"
        "LNO%_U~7gK?*X~de`OqK-_i9?pD>HVHexH4U>jcEe&niJu}yDSC50xg%7)i_R?4>Ql@hICKmY3=m+V%kRy0g}5@nCnfWst7kj;AE"
        "Nd>~xdl9}$Ro+bB^0s9h?r3M9^KIXf4tRqQuaM5uC%}l-><#Mc7y0(`7f=jWDL{i_`bK0IAQ3#3*s`Li+u9UGsztK}dAJrAa8Yh?"
        "R!s5s_JY+Q%$mLU`+$j$?mMB<EFURk{;b{-BKu*@&I1&Dz)V`O-+9xEe<~$anjni7eBWyWI%b{JH7H-?Nk9`B%#lCf1piq$$(8uf"
        "*Gg!#aaE+Rc%wy3f^J&Vh0%#Fc~iGLIDDC(GxiMJp}XPu%FdSH8>x743Fn}%H(c?(0AEU{;#xzAeP<3SlCUby>$cbmZhGK#1$MZr"
        "OR=VdkD}h)nB0DXBXj7)+7j6|lAFK14CIMLBZhu@2t%KRk~*auWp4srERX*Sj(`VrjJ%Z!0@^y5Q`w4XQ?1YQb42yN2(g7x<y`R&"
        "T||c-2XSf89Ix`mdi40i!diaPY!#Yk^7d1}AH;mGTJJoLNXEXg7v)9{bV=q6oFH@)8*0cmMEthif&L0gQ3a(UfJiBotk~HKJi4jN"
        "!v)B#K!609NoK*tp8v?Ix-?E?2qKF{$pv!QmjN%?_Dz#~4OBCasw>CDKx2q4B!SrwNsRGDjK<O95Z7yWpM&KU?~JH`<@eVyqw7Jo"
        "fHklxYrz#)cnf47l%%P(vD(S;s4DD{PO~gZ7eKq#keUr|OOaZzl}}I^<84LcbEUhcHffTBf?j0rpTD>!olVK%lYBrs{X1DL9Niv2"
        "x;uU(7mvPP%t1Sjg`u$efZC&Ks34fp7DSk1YbaBI<21JurqKPy>Ld*)X94NetPBh~MCZ#EE^+qFdDn@yN{OI{dd4pGVGoH#)g|P|"
        "de`o;($!F%#777<$!cSLO^H2tpsmY%19mr3=AIC0OxW_hX1h=@@NdzU@JkgQ2sHAmY{j(31fJm`L+^+z;DKUazf%1`+cOLv-VIrl"
        "+}OW?VAcrZC-7YggyDU9gO3zyUxq7AM)0pP;Kkp=J_=|-AQOQkV5T7~T{;W=J+VQVT)5~A+%rMr!vWR8h45w20SQugh-5uO4Z#Zd"
        "&R0U9k>$d(43N6tr(QR0zZVTCY^Kx^J6oo?IT*Uwh*nEwU(*@VSD%}wfb^m%(L&aB+Sfv$4?=JaEsT){`YGsL>??9<QHe!M<RXlT"
        "H{LoBhOBSv550g65{j&*X0?Vh4n^~OCO7DUtgawzYIA@Xf|#j9)pvU6c|jErSx}O6K6UrXNstW{0a=fH96#l}s!|Uf1-X^(o3)c<"
        "G{)KR0bC1%y3oK%I(09?_1Ya7b_yQfV;v9Zp*1{LRA#3Y!>a<D_mS)r2x8xe2|!Mta$C2<jugS^E2-@W&@`pc`N}m+S5}o4HuD<T"
        "b$ziHd#Mho?T{9#xTRo^2dErd3icNWxU#P%_0p*dNkpw1OeeAg&MT|Hbrhc#SqpV{3Sz5-IDSg@J$~E?ep@_%4T3w(i&G`Cvl;`="
        "Y6>-ygHj8uaCN(tgVH9x6e%d?YDx<1<M|%QtwN<wc%35;01LQ*MoAif5^9rv2WN|_4DaQiiU%;hr0HcUM<fV*8Or0fZhZd=l}q-W"
        "Ey@}P7gOHgjBtkiyXf@#CG~V-olHk-sMQTrUm^inPpn^v=ig;M9x!=XqS%NwqIFkf7!UaBZps+NINZd3U@uPA%6EO&H3tSRX4wS2"
        "D0>Grh}<AY(3BuRxLF|bV+{q~5Xbl1!3F`_?nunj4LB|##;*JHDq^^QEQMC{8at5povo55T8^mIAb2bLfTOSD3+jrfLy_og8KxT%"
        "VaCLGJ-tsm8F0-sM#u)MPHr2Gi*<H-lHIGv2HQ)RIqJ3&*p*_-*@7)`tX^u8u$X(bT8GCbTYPe3hkk#4JyB%l^7AMiZx3nbJ|gSx"
        "*Me?O+2^!qm*cw0JBrZM8i1!#<Dh|*yL*(!mqRpRm~Jrc=`_Er`2LTS3<A-p){}r-A<Kf7H*32gIqd-Uf@|LkMe+wCS_P0nQXf-}"
        "eX4ltg^CyCD}89o8zo!pJSIs$g2QBH3QB%1y<tt)QyjOgTQ3x8*ZCQ)>?n&nxEylEwkRP|ackm08tCAJ4u)Ob@=o7KGx^q+rO^0="
        "^f}BChiD4MT!U(Zu6rj^N^J3ja-#KFm?nMOgZmtcehja|$O5Bd9ZWUDGo(1QLzK>zF6d3yTT#ihHZy153tUw|PCwTRr9$JJXe=Yw"
        "bNX(r3gqogIjY+#f!;|2mWKsw7o~&)11L5aglj*vj(V)(;W?6Nps%PXwP{;YxLtJK0dPf>CrioU8NFf{o0sqWlAzUb9D;wKpnwDJ"
        "1iU7YM+>+9@<sIQZ0@*U=Y$v7nQP&Q5;%CbGHg&G%8hDV26j1rBW07?SgiFHcYK0uFx%bCwfz=i3nFbCa9fLhT(9`h;jnT~PXLKg"
        "`-D;n11T=ea=2liF0DGy+z7gIN?HxhHU-6DctNy36$D3A<lKGD!Z15(g)mqK$Hy((5#t2lkG9e4TEj%N`?h_Ut%9o_X|fRWe#_y5"
        "*0Wgi9O??+B$c9V;@TE~+;1q%uQk+WN?2WvuTgVk+c7!_qG>sG;i;$S%!mEdk_mTNjz^g*)h@0r?&$<~pE*Y|i#8qFJV>*|)-LB8"
        "Xlrs@=Va_hcK+Woge{P7;JiL?je<pOk*jdr7u}M7p>K@Dq6JBus*fgRzuh2!BMCX1d*A=9f<K^{!snq4741QyCK8`$A9IgbbHGRJ"
        "Wd@XccQa1>2kmF@34#uq#lFGh8vQW<Ep2*SCnj$$-@S|X_X*`<+YkawFsCj70tEgOA5oVFjVf(TMdQ3o$6%wPkePWOKW*1%jCx0P"
        "3pV<L6|G$*Hn`~*$ZV+@{*Fca%u(0+>r9@|@d0Aox}I3~9VdYyGbJ<K$;Rx^#w=fS^VuyvoyQW-LydIMg*te|92d1+Cm2~2vU~L0"
        "7+-MD3cP#N3_qi;MJd~gsLlr=_IwzWVl-R9F*fK&d~D}0IJQ6HW4<-y@AI5GV{b^=Xy4&?3>54J3i~%}1jhs#L;C-rQJ`BD_J3&T"
        "g8u)8r~r%NEI<p)?`J(EJzLw_T;pbCuvpW?Q{cmj!=D)K;$&Om%pa|{?O;O^L>E3|KLfMCmHgQ1!9#I(w&w?s3fE@>HxheBc4NQ&"
        "_WFAuH|hiWoh{!$|AdRmd`kQD(qMRO6r2?IKTt~p1QY-O00;m803iSxW4xPK0RRA$0ssIf0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7"
        "ZfS03FK=RMWn*+{c4c!eaCvP}v2KGf5Z#gZhpQ*M0*j;@RXQ|ErjAjT2t_WKV9A`3ZKUw`g_xv4tvBC$_wL?3LdYRo<sh`Wpsb)1"
        "zHCy1dVuYY!`5gmys_XKReJ{pOZg7kI0tR-JfQIb0&x)vIUyvD;z+ffu^z_8vRVE(BCmTay)q~?2nYMAA&x3phvDbaj`C-T-kNTp"
        "aRlSj=AyjIXN`kIXKVsmd`tbOIvO!no)uQ|nu9_Q8%S4K7Do)yj|Djo>Z+axaI91dFB^D+(Q}Z0)YenY_nnq>X1hjo@|3R&HsLcA"
        "nr;xA^`~crauSY0_i|^gv6Sqtoh<!$^PmQ&&*UpP79g-1<FHax!y}fUdZQGau*cfd{lAUTbe=oy@$8>*fdgfWMWpX7-VM9Hr3_vn"
        "3xpZm^Z!El4T9I&2u~Te@@AB#s{|HDNmm8C2lT{Z%Xl`=pD<+f_5)B$0|XQR000O8001EXTJIlm`40d9W;*}?C;$Keb8=%ZVrgb?"
        "V_|G(Wo>O?FLZKYX>Mt5XD@GNW^8G0WnXh<bS`jtwHs+~+{X300sn_7fg+^5qK+nMp?V9rwu}~tl|XS?6x#(cOU~}HA~`G%NgKoe"
        "z3;u5A!kVLYGtPc5-;a?^X_XX&-4B&+aeJO`$68mVL!kBcouI(d{@e}XxLU{rKoDQmKAHZg5|t{*@S(N8DFtNHew~^T^+uB@$$u+"
        "gqMw|PGynpSP!FSysq0^(Ec(l3fNv&qN#X_KWo`mv4BODVEmq^05=l=??$rQA7(6<iO5*luCg@dO)85Si(!a~M-jL9&JZ(^m$Kp+"
        "<4q$98jU$%0AM`NL)=na(7@Jm*S&0cz0J~9_x_(+7Trt5i$vxxU#u&cvywNslIgjh;8oAGZJL;2{Qa$USR<?0tgHE20BW@aUg~Ka"
        "iexE@@sg0X_6zStv`#aDuxaTkUCWqf8<FQcOnAe?SY{cAU1ra%D0y1F)uU6}i2rG^2}=pA-I<O4%NGouRhwo>WY$ERO15P^Q**Lx"
        "(mefDRBAZNd6BL~-RSAI1dK$az($$i6@jQ=5FrFe#mg6oSTm$4O41GN^^wKFm4cv#*xX1cthfB~yYKx#O+Tc~mSmtdg~&tSt5#kB"
        "n#Ss;5<J(C@Ilp$w{3C9(t@R+pnk^lRl=|I(h&Za_!sAw-?9^i{{%C(^1Q$T(IXWq>U;vI@c~_KtHLbUibqY7MQL2Nz)T_hd!Dtz"
        "fKJyet<$1zcoFNVnbMl@BPoK7gFb~;g^6fDzx^Ntz=kc8(2_K6CZK$}6T7P(E8t2V5TjsxhD^$lS?Vl&tNq~ffm5=yo;qbQaexDR"
        "d58~fnTZLYH10nI*%b{9_0k}Ns@F}_J1MR9<ZXs(uHON@M(E#FTY%!4nONqaiRmiM(q^Z}PRv|)L<C5J`<#2W8Ais15<d4A9$JVd"
        "x{?x&fHYkn7=W_YGP@T>=(r}#?_dgyp#tAm&p%ekKJf@LiR2x<>c#7{03SJ5fX~?4>ybPCMC2{L*<^CXv#4%F`OWg`{QKmy*9nRC"
        "hXffE!WIorYr)=wIbYM{bydmAUwd!+K%#d286>i9LDm~4??v@cfzYnlCj$K03pz$`M31KLg+N-9^6b=OuauVs%?CTP*J~!|A=tok"
        "@TTBx=j>CDXaS>FtQ^=6cuxo#n^v~90bm>lZkVwW^@?;O#DOI0;oBR4MI{L6zikRq0_VsRHK$Fd^W%QMeTGiRR}`NykE;+`hgLA~"
        "f{v(NCVApg<aKK5Sq9w=h9uSq-GCniGizVx+i8V&<><Fsfu1O;iVaLEue1>jzN;<11FqaPTUqGUDtO}ZiIs7ia0e<5xS%7RWqyGC"
        "v{s~b)a3^-DKb#G#d7AbSS?3J`<h!b)-hLQ9F;mEEE}V5APJ5^bii0;uE^D)3ER|I_F{Ora0ZtS?Y5^t&=a1cmkv@Ip~BS&8ltmE"
        "EY%5I(t%kcMA>}K4!N64&oW}ehJ`l>@U^!D*6fS=maY+QdXMI9ui9pPh-sD=gk<|sBOV(cN5Z7d%i1?cKoEkQv}op+5RF9vDIcin"
        "+-sZl>7P2e@#;=gQwDA)76t*FnJ{j&Y=7GBW)mti8xi6Np(pqD`pvt4JwYqse+;D{Tjk_B?;k2}&wi)fBv7Ng^S%`$Z~YI9iONyi"
        "e!Afmnk^G`%u7hmA@D#FQHdB#YZp2$8qPuv`gE`xaM7umVH0%$Gl200f!9*&!!Z{Xrv^V#!{+}_7<N$^>LJM%olfohI(WHyfSIOJ"
        "gOQ*+{!#QGC;t^%+4}_GhgP(JRKdUph2h1{w+zCv8I3%d4DM0@7Wlq&EXROTV8$uI9wkem2r5v58IZ*&KZiLe5;0Rt6RPvr8hpJ-"
        "!rs~yfqVEobT3b{8ko~<0+siT!zYSI`f`Xgw@|PN)owUXYbu<t7&%8QfK}0Dk_C2uiK!8l$@dqk!l81x689;jC0(T*di`NzIu?Tf"
        "vm{dNp<^W^9--_{b8sMMXjNy8yb}f5<XLB(0JvDEn~u29@>X9LiAItc1tgNub0gBApgTx^PbF;z9Zi6Cip6JIb|bpoq*JGw$7Yce"
        "$)jb&qu8&k(nx)nEi8}r_GV*`JRw_y&tI>Bf!-+hR4Ep|bhbC)k8)l{nN%H{xd(YEj6H~Svu)}K`$KkF^40o+`Cg)>3>a!$tvtqC"
        "E<9TmyrrXTdlktBd|2gN)tXj7%W|Jo#h>xyp%xmcFzV*SZAAfIsbeQ8O_S@~@ySG?2-3U&E)Y#Pu3=d-?p{=4+B@rx1zr9%Egcrk"
        "_7F}*4pbnv(7%igy_UwZ#n$3LO-Tc76&O@GTJB`cL<(3IEZ1o}4MW0i-qpOf9NHnRQ@l$tTJmBed^`CF0t{-PvFx0^o||)CD_*PJ"
        "2QBpP1;s(jEP(qMEi3Ja6vWWY4@Z@U>m;11V;uyp9(y{9PhJ1g$868|g6Rr&1}Y1Jdyv)L+#?5(Kwyi7at+kC!+<R+Hws|#qQ=8J"
        "u>&tjOvh2n6+XwUYwWPiHPRiU=a#eXPcdDv!C_dTQa?&#iB)V3<|us1oz(jl5^0UPT3m)SHUu__z0*XYZB{lxLFThffXW=mMHiBy"
        "$Oz}>(uhwPV&^=v$yBmcbOwXOox($8wH~Hia^RHgPG_Mv->wB=mh09+ZCVT6QmX;6?9HN^?z#?3SH;jGCl)Db_00#Rt`T{i5r+<{"
        "+}mH?z0umPoR-$R+EF~~sCv{1nZj>SZo$&N(h}J3#LXBgGg(8aN#zw}b>v0BA)n0H3Ux@;N*X^ySC@6pb-H4_s8y?lc!=N0u;CfD"
        "t|3XpUFYHXjGgPb%1%0u)K4I&q7b8eD?mdz8T11i`6+d!S-``+1@VIp!a^Nzr4M>#!CMAlBYn+~F=*K~W!o6e__S|g8Lw*rfYs^{"
        "Wq?HwMuZz#i1B*RM4W^KC-W71(i#eAut?v65lj)=Yv3(t7v$_(2X8PvM+kU{mz)9^hA<qpGmlctIm3cn(G+HUC9)dL;Pm4BJP4sS"
        "@q>x2BzyL1ZqwTLSJ9<Fa(kjqnA%M$^J;D{pw2;NtBs|GzR}L!$dPjj(7~|wH^YXygi;7EmtZHWp&qO5VQr#ebPO_gB_5JQ-AbhN"
        "OfQ3)yWbkAF<PW2jjF+zqr0E&`>!UF>vaanGn(o|D}*RE-3@X7?AwQ&KT^Q2=O9BpCQ2D^VL-q3n%@oaIBxScQ_1_cdeFJZn_r64"
        "@ASk2uX0F*)iGu;&e9T6_9~xWgx$g5F(Pp<s@?q3tfQXza7>6~e(^^`ELFu3ro3~sotg1basyMfJ%%Y*my+8PT^6WvWiw^n!&$j<"
        "4m#HHZKm1fx=WiuGmEa7MNhz9vx^=oI5mfZm<^sE(>5`=prHZNp9cP|?d*>%02i3N6O@JeDn<-+9FEx10u0X&o_zz70C0uK%Vwtz"
        "I77$~gPG1y;eZmbIa6?Z+nKn#a<H|;O5o@s%Umtd2v#sOig&hjxAXLs)1VV`Yd0?$&oF)2Vc((wO7N3|CSCIwuchLxlmM>~5A61r"
        "o9O1vhil~4?e)*Ue7KI@-~RYvqVw{s%QvSxuuy#gt0VA4FJS2~uU~vvd~&M5?-ggF>ONR>U-Vaj1B=4ohDK$f(-&<fk)o0CO~Uhk"
        "IyYchHFr$IbPA$G#C$iWWCHLL5K-puc~%>a;th~>U|0W7e!E~&paJ5+u?U+=_Bhl=e%ZaHz~|&--{lXf?f_4^D=K#oSVf6H{2?8~"
        "{iBxI)&Ay%Zg5y2K3ilwQ%NqpE?i;CI|rM3Z2Ym32Rf&=Dstu&)EG61${=ua|6BUU$1nkD$+m?EDs)wCbpfvE+Xw2SyLWWw7@X0m"
        ">))#CzcNh$Q}wIKZHku`$o|E$@5for%WgyI=EK2@4)fSiex!keaqGCj%pn!(`8V7&x(hjU!`ZuSC%oF}ti^+_58y1GxCe7lvTntk"
        "jR!BsMmTndW5iM2>BBd0F8hwTa<uZ$krO4yt{rv8Yg_oFdwrwH?b5gkr7pHlFM76<nS{*_N`LH%Vi9O}KdsUcz(Da>3Vs*mA_u*t"
        "BC>X?nM`izD}??F3!~-4tW5LuVT)@f5q{n0l@zJ<B4t!Kt#YP8FURevvA?KpnA>_A);9ylP$pcs3wpq24Zs~8Z=Mx+>buDk^!ZPD"
        "AoGkp5!8?_BE4%GkT`jM$CfTDBy~&UN=B4<3R}DI!Tf>d`!d;1OD8&2UnGG&Yy(qCW5UVZz8D^IH9FQQwhuGomBLUTHsGAYoRF8L"
        "C=!2E@h?b#3pDLVb$3ZW>03Yrb{RN*7%1N&Qx0Rz`(#N>Kvl;9vpE|pw<X_|sS5s+O{ANN;uMeQ6O{)bK|VMYecywF|8$21okAlc"
        "Oyq|x_+0RUT12XA<EodEQ(U$q!UnjTeoMDuZwuv!_Q`WvCtuMvIovYos;$YE$+%rYL?6>KvGHOj%vYe!Y{h`oXvO0@sHhTusPmn;"
        "o@}^GLUy6ym(>c0E^qB*nkT8yF|7>owXayLQiduOnzdxK9I1T|PS>Z{)|`^sCJ5=<FJ)C5JgwV1-F9^)_#v(F>%y~QEXfXzBTg!g"
        "O~PLkVcVox9a7zl8>o1WoN}eBsrRS;2h<Wbj`Z=cE*iuMR|k3erW&w!4P#wFO^aVp%&C`w4H)K62J%<zA^;7sdjeE^I!yw~M=osN"
        "a8&(Am`HPfZo4h#Zg9dfX=34JIvzV+lar3Re&3)Dv>9{gTlD@jf!So5GwD=u%KvhAix$`E;mq}^gF<2?D<}7<<ioh6^p5)qk(QbY"
        "$W2-%tI$aIKRf3$T1DTC4SJx;;|!NNBB0jjv~-924AaQ)U>}*n!0Aobub5qvHcS^RnweoKOMigfFz1sG3wMS3Fp_>vrMj;CesA>&"
        "F^@kVBLQN7B?NTu6%KZR?Vj!U7H2*Ryhjuf9R^f9fD6WNH}GOVT@gN3k%76tt}eQDCfI{BxjZ@f!}&;;O&z^7i5eU>ytXujiT;BQ"
        "?oE_olYil%&Y{2MpgPYk@$ek*bKbv;!3Pqifjod?h6f1lmulbfosz0m9U98V4sZY8BMhIZu7P>>bq^d#?3u$taL?4C7Q>jzV>$}v"
        "8Q9n0^%)hkZU-upQGxF_v!4T>R!m`NiT_E_A18K=_b^z^l*|U=IYttDxjwk|>L)xru|?1Bz5VgSPd{9Lyng3>`A6_ih90ZyHgq4("
        "Dv>1-<gXEV%_~a*oKNn1F*x7y+fmH3`Y&{7<y#}cWQWGOULp_!$GO@yqIUnp;gI8e+F85#&h+uv|EcbuJl!n&%?@9;1x0NT_X=G0"
        "7QYvcs<r?OC^epzM|9j@SbRk%{8?S_gMY`MnKP00jtkQviy#^KA5cpJ1QY-O00;m803iUhHXr5_2mk<P8~^|#0001Ua$_%IX=ZO@"
        "VQgn*ZEayMbaG*7ZfS03FK};cX=8aVaCx;EO>g5i_8uVrfmJWk)T3!yw1@2YFpEur0n#>TXLb*UAutx5up?0_NhOoX`oHgcq)3sp"
        "Y^Ok1mssNa{^m=TWmlj3hHtngTiWvGiS!-3J`pNvQZvnD%LP}OZ%Ef`-tsR@k~`TAy&~IA647aP+jXDF?>~Jgvn)G5J3r%X-$_k2"
        "UH{}?TB`4ZGu_F}eJKQ?iU@%RTiLbbmT$WaZSGjx(o#zbti@f~vbK{?RWG|+Mn(cR_;TI-_U@VsM&$yyi7meyB(rdp!$#WYXEob`"
        "Hd{J0+MgicmN!i0#pU@KfoIdTZ70Z*Jll7CvIiy=??iS<GO9Hb8eXcVleDsysva(~MF20U_yoWgKZYM^)8AA0`u%s|*R5pi3#%SG"
        "St~gFAO|*c+&66I5a=oVp8e5rp`8Rf`^Ra9s*In|S)+%!AbF;q+Lmd_H`xNjo8Fu)@EQFKL<DWwC3(&o2`s;3uYUlfM=IN)uassz"
        "O8>*o1_VG_4oqyinn|QzGyAjHO~m=u$+KKW5(ENFLae}q{7$rB|7?vC9T#OH;Y_iz%+|4Y@p<(MBekp<>+wf!qo7#CP?^OC>h?!^"
        "Y@(`~guFbUU5aPb191BzI8J=Ia#pN50dgdYoK|8|ASM)o>?`pTZB!I@UfyT#)sum-i=$IAJxJjdn=<qPIM-COAVp1z<@V2lw7{=f"
        "{?ZBN!qgdgEL-Ir)49{Mupm>QL9xQnd_EZbi^Mj&AQnL+#mK*DGq5WuJDF!6#V64{3gV=FJ(f<(tqk~w!nt}<kgt|UN(zBo3ejx7"
        "TO^zEM`imJ#3u%hcoqcfrK|4`7lbRKh91i)s}D&Vm8b$y(1=)!QUQZ%6~P@9^FKiM&CT_dw;>UX)rwG}SWhLy0;u7_qIUcOS@l!T"
        "@=y#@ehV)F8o&#>vZrhEmRuYa%NPn_NjoS_pzTl~p}6J`Sll$bV=|N?pNJWziVLNw*u<jQCQvL$4kQ+2+jLYH#e{0xscDrZS6i@$"
        "b_7-<@;U}j4$0dk`PKpyr}Z;PUO~&ww0)<k*N}bg*#_;uB>9x~yQG9inL`-V4al^aCa^2We?yBs2c<ow8RVv=sKAh;RFnDfV`%8H"
        "MlGQE6((c60!hrK_09kXqRmV45)0qhEVGil@f=9qIUjHC(?a3e=K>YdkPHf|mGkka)Vc&WwbRUkPXqIhExO<#hif&2x?C)hZ&7^?"
        "0tzgdL@yGKf^|Md8)*(2X#z0m-+Q*%X77bbX>$FW(b-HFyD|~QWVRJG&E`?pw||A#h%>_vc$vQ9m|%Okj6^vU<Oz32s2qbnB#2^1"
        "hd6!y0YQm%d7km%;QE&p;x!84>bYcp4qP%Uowv{_8W@X5hM`>zJ=1<81tBb;OUT?HK&nnzYkDmT7%QP5pGwVH<+O2jF^wY0q9l{j"
        "L`susUgw{Uizr+hAmTQl0fSs(D#)h!?u$=g>g7Q%9P)Tjfv`BH_1)ty;38ALud&tYr_=iU?#Md1ZH-dIks!Wkq?vJ`KAx7&-jz<v"
        "_D&rVj*<o_2&kl%RwASu<bunX9cYZvU3l4VPK=igeWL{w47UuNc<8nlvsh+0R6!JDD__PlKu~i5XK2i&#m0txk4r3=J+P~VSfsbm"
        ")Up<=RB)xm(Pp{VuiqqtNi-iDT_>B~4!o%+BiVNFH<%QBCpimld8_s5iyV&4Q`a2ldc~Nq4al|VuML&D!cnfpWrBiRcs=040mOsx"
        "^PWnGr5krlx?7A_wE)JOW~k9>zz6D?D$^e6X`+zI46)W4Wy&)z5>H~5_~X)5PB1C06Awb1LE{`)90+ypu?xc;6hEC*cE+W*5Neac"
        "k5qvNhL%Y);Oa-dVNt##?UL4Y9&t^&Hhe81G^JLLT;F?-1=S5`j<cPOw{-GZ&)vGoGIa9<0Nv#<Q^(1iGf~^qJS)uw=N*QQ{rS(V"
        "NE60$bs=|(=NONF3r`o@Ay1Vy<e3TMqwu$=z{TEIm*c$gkEv++R>h@?tNWOYD<jmpq<s(C<arYD&i#Hrvs%a5@xr{l+>0W2MNWs#"
        "{Yl43idgCKd;)wFehi=H;VDb^=9JmoL-~%ep7GCB+e0^<JY#fnUWrp}oVu5L`deP<X3Y5~7U6l{7#X5xm$q`v92+a(u4H8=)C!!9"
        "ZiqBpAH=Ni+@K2^0t-^T4Bf14oZc&Zodv^*vU(wGvi+G5qJtDQ*w}^lI%p^`OFm=mD~xLW)TYA%Z`KxZB|pBq`88c;j~BB|_rP$f"
        "Y35RSiqWw}TimHfC;hh4h@X#}y`)KxBv!#icy2Z5HiIi0L#f>Y*NkB9!yaVfm=XM_SF%ZUbmh?IeTmJg$pdeUgvJ<N?=4PiyvZ4S"
        "+oYcr+%)>qd_GKOw@1fZEvl{TzA)UzyA+K!RC@~CTwfhT!By41m4HsApTLQ@DYXMi_5MQzbKiX@vuQuJa%;wQdQC8kip(vrvi+4V"
        "KB%{CJ$#BIJXlZR<p1Mhr@gABf^V79V?@0?lE(6LG?3y-(s0LS6`kz6_x~1vf6-V&Z${oR^UO5`GZ9^&a&O56K}ODUSYzNd*(4mR"
        "76#ny%tR*yyQ9&r%<xP8@8U)GV%Xv}4)FnNzr~p~PGAs8{e6~;{G$AJL2@J8H{_y#dxc+|Y#mJ#l>nGDF<Zzdj^2CZmFE(gM!V_6"
        "#ZiEZXi-QS@bk0(0Z>Z=1QY-O00;m803iTufR+o10RRAr0ssIe0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FK}{iW@&6`Zf7oV"
        "d2NzGYurE#h3^Lc2jNo!8@te(4}~0(9!i(8O>+&(SmPZsGjcU@H``xd&8C}@5}$-^z4!D<UDw@x_&7o$EUVB2dz9&bLa+$?5UGbP"
        "!~O31D^SspPzgc-6|!mqEP(%dg~}SW@4BvS+QthfcnqE$#+g#W-f{L}hL9E(g`4MvBqYlEeIfVW|HYyA9H(=~t(lt9mUTxs9dATf"
        "s#Duc<j;X0=vDWknDF?Y(FQDr(z8A&KT$3tgsdA-l$qYVyM(JxFa@V$+kibb*6mvTM>tTJ_gV&C3`zJ+GN9A&3pH>qL!s!ezT6K^"
        "xD+d@W1I$~njjjs?2dg6T}5|vn(Rz{V^fd|#dIuE%>t3~tb5~}4R?%2M_FJYd@4|AK{3U6gi7y4lDwyy##U$_6DdjQ;qJij`P&b0"
        "xS$If$P{hMnbnq9)@18bhDU2+jx_bJGrtsqq0qYvxTpuzZJ^uY$u#jqvGSu?PC)9t2>o0`JrQr&jn#_r@4iBw`|h%B{s2%*0|XQR"
        "000O8001EX=*qR9zz6^UFC72?D*ylhb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@PPaBp*IbZ>HbUvp-3E^v9JS>12rHWGhdVE+eJ"
        "_ae1gUmUc1*9CkJcgO<w&|=XIiUK}EU}VZBqOT(<$8M1Sy)*n0DO=8NtA|7)Im4OZkG~nxNs|1+`c87yi4$SRz7vY5?vb_pcdB^T"
        "654J_LshwF+jo6e^YTPKf4nDECkU-;BKeV&U0d<pAhdv7BuVo2&D%GX=o->fwXgXD;f>b(2|j(_1FyQ0);rcTbYXP3ad*Fcy8Gq7"
        "Ki&O&Uwr;__n*%*5=zigNhfq~%w}8gh;ngvq)lJ5OQKt<=%TBtnzyWwm2w)=w{HkMy#NJ*6>Q5DZ+C@YWhb^WH@&j2xti-!^r2=&"
        "1wBmkg0~7fvRM<v7<y}gV02r^ZV)9aME4|TMWZ#c!gUA22PSyM*|xAFn7E_^BF9^>@=_Eka4%@SDsB;BB^?l0VcLtP+p;<*TUODb"
        "R)rHQdeK$9W)iXR_6>A9%z@WFCN{QqQ6L&Q#GaKoVQmR|1?sgg-}=U)<_%Y??{9v5@B3}rKbJq=!fJdl`gnyZ7dxiXM07PvGV;xe"
        "B-+Hpj?wc{E*bfi)&si}q7!Mt4)CD^?^eG+w6tB|Ag)o887&A@c2s1A6Rs2cnQTn2vzu=vj8FqWHnnXF^FOr`IiXts2ZVsr+AObJ"
        "-HU-`8G3^qRzeU~gVAZ`T@5<}=U^S09xY49Hhh2-)F4{Ywe^WmGPD{}WXr3HR6`AG2k6)lwP%F4Qi-9|I2@#Fb#_?O7yoOMc!uvx"
        "Fal{85PYoR=*1;#|D1VX1egIcTceEPhDk|x41>RZHc0qDl0h<&<ispitDwnD{<85DQ33O<9K;s1@9R@h?kRwP0q;=7f|$BVa*y-X"
        "M+(lFCEc->8S*U5EP7QD(wa@wp0|(Sj}b9SZGgbp`m2qA2n3!|DLK#<_!zmXS4U9+`=jWQ{*%qiCLfhe{1`hfJ2V}2(X~AC&4f7G"
        "7tT++FU*LG+d}Khf+$_ngZ&EuIN^Poy{?W|UgP=$saczjre+&tk@5Q~k=S82$`9SpZmBpq+Az&vv!84-75T971_Akm0fXc`eywk`"
        "QMcHN<E&!%GH>g5OFdwD^Zu&20W(M8v1Ff&;Xy+KUdzb8VtU58ag6gUBk#x(21R`RrM4IwI>>uf-90%ggkgxwH9_jRhqjPt`8L_)"
        "Ara?6^l}@eY|K@zJ_wDl#P2EysB?U|04d9^sLRMjNui#yt>5NxA_x+g={V0~P9IKcn{RJ!Zv6Pe*yyZaY`a?iBOmjhD_kJm*NO?;"
        "95Z!KKq1reBt=JI%b2EBf|5NgA1h=NL7%K0%Nu;6-EA=oC1fgW64tX7>eOQ{UIEJo@zoSm9F~bLJO$9Yb0E8lllwiFL<}wP5eWQ$"
        "-~#whOe<c90vglsBV>`9?I@HqR{g>O>Tl$uVC;7WNe9wf&37CK3{uG+3m;mFPD)=8>#|3Z2L{G6vJ>6VgJNAcVqh0e<G-n7SY}DG"
        "rw0a{3w0LyhlxwlhLHo8b{bPc^z|Se9xN`U)utf@i}?J4Ff}-+2(UX0Nde7F0KixnEv$fC`$C%xc-Mh|=xTMrU_ZtBTxJ8)&eWdc"
        "%(*W9)5#uioEmJ|*;#<coNEs@aF$M16MFeB-ZSw1!}8Xqdvlb@-AggTe+?vn(bFjdQPw>LI16H-#Ay;49A3gni9_9oHOJWjW->lx"
        "gV~CA`d|foxyMgE({EAWhq+2n*Ntbbrg;1raQ>z0Y_zkP@iDVSUo#Rh<;9CRZ*q)^3#zlY$wC0jCf05-DFg%9v1Cwej${IE6NWF7"
        "wFFL-Q2Van@F!9|;PNyMX=@{Zn`Y9ZdC>Ic<R$~yE+eiUB?A7(Q_cZrzjQ2(sN3?rP6hZUj}l1mo(Cz!zeVir4_@$6G!*J@Ce;-d"
        "(8muibd3o7*$!Sp&4DLZC-L&HvCZUA6fC?xI%)XpO^`*Nmr$iCX?a#Jf~znzgN*E<*8`~_9hWI&UuH@X8HhPiOFUY`1Z?Nj6Y6qE"
        "zPzcvns791W=RL1pt!m8xW$+s;V|*4j;|n(K<^xZoS&6n<qmwZS1IL-Sp%O}nFAjKxPUZF!z@?e<1y&~;Q(@8eEFf*>!>?8#GMRZ"
        "{2lp*RthT+6t)E<)-|qH3ij>us5Jp>1ApP6GeDp%Z3;ZRV}TchJ~5nTj~5^Yyx+CFgo8(*K$lm=zN@$0pdxWwQ5yRS6E{#}*)|;M"
        "h}e{KXA%>hz9I<+1{l8<lhegn)fyh96LW%@PHJAFZ~$pThvx2}IocD@KetDcXdA%5zp^coky=eT0Q#PtDaHs--?^XKOzJycLWM41"
        "TGQk@%HGP6^>GozRY&!RQS~T>8gVn(yfF9Si$s@^Xk?o`^7I0ZnJub7h4D1c=UeyDT%$a0&bx5>N1YuapyY0<@eQDkr+}{2O)c%^"
        "zEg$V)7u~3W0Qt|G9iwa^YS#VX2<?>Bw{eaD{O?^*}e88WvtI#T&GdQb8=inN9?032#WBqeO%b<5PdD+SBWdu$eB3mzGx}qapAq9"
        "b@nCZsdW;%yU~n%=CB(aUoeEo%Of!5Uj2A;fLH6`HA>~;XfDdbvd*97?HiC=Ke!#}cUF9b(9mNc73(FQxZSlv-<5l4u%EOnhi0ht"
        "<=pD~-1#Dh2D}wrv$A!+O9vHgczucsFi#|S{~Fxu0)uEjJcYp78`Vu~t3;lfhAF|zWR%g|qc|%ZNZce(y7bkO2^}TlvdsPtv6}y>"
        "A<fptmyIjrF8x-K+FvY8o3Htr9s>5jAsd}0IgN^P4*)Cy4Ej2?g@Fe)S>$P6W5*s6W@Bzz1NU1v^M~V%!I-l2OT^_=(a9#AL36af"
        "eN<ddH81hDu+X0T4iw%b2WXulHbyNzx)L|=4Nr(sRD<>r#QrV3I5u!L|H5nM5ZB?y)XdTQRpk409|Z5?z4RYYO9KQH000080000X"
        "0NVjf-y{M603QSZ03-ka0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7raa$#w1WpXZXd4*F=Yui8&y&L==2AvX6Wz(h?K}t(vmmFe3"
        "V(3wfcW3RT`=RbfNJ4*nXJy$g#i2S#tKGLV^WMCvs%ky|*n-r7M1rT~8f4JV6reXKXy=Wj(7+QBNH5}S{rK%}b+`K7SlLunHJwZ+"
        "DtHSL6AFVdQUq;#AJS=Sg!GtpMsE(s5Bz()zR}7HG+VM38{wVO+XHU#(o@iuTzcwBoWn<fN^zlw8%x5J1Fd{|B$Tg<5PT?^rjxJ7"
        "nl3oZ8@cOs$aFcK410zB&5*mDPJkan7@T(7E?}Y-l<}D0H>|uPxE?-xoYf-uZi7kex~Sh6s@+-OHlWnZq(dRH%?6Y9u8WEK`0gEx"
        "FaJUG+&UdOQVTJKvj|d&gA^Lu*LP8gQ0yplfqrGSgKS~ry<x}_O{Beok`!<p?Yg~Vx~S5{{(9v0wEwiL9$e5_bXrkN?a=ia_8o3d"
        "g4f`4>T@bl+tC`A@Zrl)-PjGb+cKBNw#?WoZKwbunPS|bCfx8|j{z+u3b7pDEXtc{iJ!e^{hC6eM7y1Tn%&N?0G)*U`PJ-x{uykJ"
        "2{r_w*oGI%2Lj(GZMmsoR;U1bK(ZwoSnqfwNMSZ<6q&0AM`d;lX2&@r;6c;|R@oTXy|OxgRX9owmM}6iIWj8=j->Ys<`?O#wMY2}"
        "G|zHXL-nJ?EaleV*kjL}z)LbUJS>*$`b249e-ekwiP=&!rsTQ!15ir?1QY-O00;m803iS%Es_VN4gdh#DgXcn0000?Nkc_WQ$^ic"
        "TXWmS6@Jf5{=*Ju+L2};+HsS#NuFd&j@dethm`Bd;}TeqTLV~R7ZAn%`ku?)KvJ?heXGZlSP=H?*>m~Mcg{kHFZ{XgWw}%0y)ISf"
        ")VJR}^Xfm9b=q{|)%;~4{w}*w+JkuY^5yqWui6bm|MQC%`~7|{X`{KZ+ZPRQ=3adJ4K2ODUH!He*UKC6=5BejxL@2Y*W&HnN_<@3"
        "7Gib#VRd)&@eMvM=&hT@`hK-|{Slvd5#P<ljjDC02hh;X{Zsfe^Cz1Lx06jHS|vL%z=;QCTPI{!iPCfxuM@SgVsxqyR`u3YV~LN8"
        "APZlvv>UA6jQ9&7ov3hERf)|(tX0WUzJm>{8Miy}lQ1>>py54Jj;-p3Q`L+;(XBN7!Rqa95N6*gD?lq)s)vJ+<FGSU|C@FUg3c`&"
        "c5)D~@779K*=^}1Z;OczYAYLYOVUsD7&|;A9YG0MlANFj*aF@SatyrTmDI}d?qJu!nx+uas_;WYiWhhyd^&a&EGtdhnl6a)Z;HJh"
        "b}WduoQt={k|zCVdjs*4j2m4aeKqr9W^@`SF7zd@HG5?XaH9opp&Gj5e-~mfqLko$d@qRM-;oupXk{n23cZLsyRqDP)e5oSDY5{B"
        "0c|Wv_{2JUje!MXE;Q&uUUEC#qo`WfVA5XM5@lcf@bXWWw4VVJvGt>*ac~3J9z6lhvdRUKAakQSFhr@rp%d|$rn&2XH{(oPz-s)L"
        "on7YMhrei;2VIRQ&5Aruf&}#l8f%BTfbOk!j>4Zpj#GlXb0+H6uwe-~3HjC@2VHMft*iwB^xK*Y{~0%FO{L*{lCms%T6bkLl0hIV"
        "MP~-l=vJcw;817kVULmKXh(2P1x^UbNRomaexnGGQ0whz=>>34qb8VgxA_-D#i{zTJMc5`%h*u5)Yi1{%W@|>&^+b~L`diGRT+Yd"
        "o;3c)T1de*B6-F1kU_}NSs{&jjR|H*J%83M1O=P}9#2j^0W~=81B1ImL7dF3s<aftp`V`pld+#qfMO31No#^9jJuQ)x(i213Bd;R"
        "XKQ5z=<*<SBR5UJgbc0<#955SQu?rx2`d8Xfi2+e2zzWVcw3Wk<uG9NAu|PigNzF}gL;&04|~H(z<r2%UWu<>_q~F>A0fLMv%kzu"
        "aigq$0Bb%dfmU&|<5<9L&Kts8Ac$f^1m!o<p%*(!&I&idWPm_o2*bU|ZI~MSoi2AdLV(u>s1T5>Ry}C)DaJI|$LEbuU}s~)Pawnl"
        "JI}5l4eOwC5H#cj3EP>5as<}tt?po(Q{SH=b%d{a0ttmUHlep5Mz(h$$@dk-s#fZhu&S4qLI!O^Cux;c%>k04`%K2$K-gfwbh1^K"
        "!TB1{$=0%@0#f8fG+OpVEwqp_b?*H)i1VHaob~*1Lc|=-w~S`=2^gwvR0X9?Tto3(dBze&8MY8DgBPAUK#^t0fF;4e-c3YwH*Nq<"
        "Je~x4LUBgQlY%}kXeWaB6Y>$gP4)2EQj!xIglpO#Lw}>dpfy<I8ItpLj6lrdsb)bKbAkxhuu3(Mj@AH37wC$OY$%}i78iCzQO3@@"
        "Kwv^lEU%KW&{l&>i9=3s#j|B9!CAfw{7qT`O6mqzH$Xr^TvpE_V{!*KsMbxO9g6a(u(*^|Lj5Zo1y~oE(?n91TfWF)WfBaT$<cPe"
        "iObOu51=hu!cfmSKM@9IRZBevLrjkw0uxT-dR>lYbdbuew4V_SZ6Z;Tn<=Na9n}PgN%Rz1_iRWadd-$#bSd(@ozKoC^bsSEM-S=v"
        "WoDF{5izqp+E?sApN)dR0(z+k-$Czuqbwtx`eTGZYH;JyfYrI2BR|e^F+kI+x%d?c81DQg8X<5nu^zef`mjCQ1?5SXgSCPhDl#hx"
        "#1zn*2p=&AAd+xWAcMUcz&tS$0S=pLuaS>*rn{mLaByt=@d~(+-D1yW4zd{zSG85}lLiX-U`oXI(`xTKNZc}XYOn+{uE&@@0ly69"
        "`*8z{!IluTeIp_2;}cMu>#(CIo_gdx-=sB&2u<|*WY06zp1>yf=`Wd|K1jsk{|_f!z&h0rm~hak41q|2ijF&}OVRUjGbaMU2WdO`"
        "pok4dm8cgtb&Z4&s)K3(mHEE_GR6+<pNI^eR(twL$P*8TW?)x^eP!P_*sPikoKL1kJoCzxjnrUtem6fKn3AOACXQ&@K{h)lt)^tH"
        "EdWwzgp`hTF-L+6cL{y8>6EXQfQdj_VjDo~ju*r;afj?{F`R(O;8Z=o#~%F=>fl@~Y7B$aLODP$jPK~0L2p?Jxs~{90zTjSU!+=="
        "n7B30U6IxBB&8XVW%17t08%t^??ye~X&bfW>Ht;?>Svtc2r8cgJt`Kgy7RqyCa6p<I)sx2ue8KzK@Rh^X-Gy|6<S%JHK8?0`R2=E"
        "U|w7`FkjJ!F=nI-w621uAqu03K^(T2Re*-?=VGPuLDQUeZ{;DO-7zo$HtN8CrV#ZE8zj#lzl7aJ06q#RWIgbIlPdPK@8%l+1iBTe"
        "e<G`7LDnk8?ygM(ofH#*0B1j^I(#87`M4uQ-4=C56}d+N2WvPd0$I+JQ)7vLPEISSavV1wzi^q2+itRLc@C7QH}+pR1mGEw#c+a_"
        "Lw6W6+<CfO0|dSp;V2Z_|1H@>v7qD4%WQM221vOg5bbOZKph1rb$r5n@3?0QjskNoRnNjldx3CSDP&khPB|!Ihm?T+fIP}LqtOZ0"
        ";-AFw7*>N2Qm-;ZfT0B8NztgBgLcTu8XxAgnjf34x<oXKvEbWA^!jXhcfWXZJA;IL9LVmNiGCYo>G@W9zGc90CUs8iOK!_$h1OMq"
        "FJ(n7YKlO0*0_jO5+}*I9FKN{7km~vMe#M(&E=do^jYgt*nyBnNr#Qt<VwJ=NeK)L3by?@C@O=N$>7OK)A(}FYVk`B^pkL#WcqZX"
        "C3Kz82W#p!Rmjr~j4jTXS_WR2&CPsAamG^haoP~efcBAH1`=&`h37m(4|O>6gN_^7hmz33-tV}9Lr^<o>&)%MFt`_wlSAk{QtyjA"
        "=13u*X@p{j$sIr}DYB~Y-xeFCJT!ALL3?kduStr6UEshQlVedUhEtNNQe8E+f#gmiCm;<skHM)YFh^Dnvk)-F*~FsR7c^VU73_Et"
        "h-{pv??}!XD)l17a_LflS;EmIE%yjY@rTLP#wnjhk~?9=XGniGE<AHh#d5OiDN~=(r^s@sre5>#L<f@({9;~_K-@c@5GEB)uE-{3"
        "I`VWJRTKBdfSOKtVuN$kJskPv2kMgiodfr}i8H#n_}D>}b>uts2z!+p8)%Z5UB{$J$>ErEX4A23EcWCmENc&N&*P~O6T?kD-Ttq-"
        "I?u>R(JXRU22<TC-hHxwr5SKlye^{3voYMyVH$6#HN^Tt%DEBBgj1E`t0&Bf+%tYFCLla|1yjaeY76}!CBeaG3pI1<QI$FL6COvS"
        "Shbawud<F?mU!jxBOqj8DGt#x<KC4aKpmK-XLnF&et(h294FqG$QEaDk@w&rP1y&$Iq*M%T0Wln<q*I@=@>Gp&9&-}QTzKUEcgyw"
        "gC(C_3(UZ@cu|h}fUSX<lyFGzkkqW;6!>KJ8*(eS&)11FwIP`6uXAyu9kpF}Z(NH{z*fOHhnNvj+s%PHVCvqnJ<EWB+(~UunyeMc"
        "aUKm^()I$?#nHn_kDI@agVBkXFL6=^#b9>57K`;vyuMy9*3rbD7WeP&KHiH@*Q?d_@_uo<7I&-s8-csG;(GbJ_{U;-1Dr_n9rPm("
        "UR`z=O;}T9BiNKrG=r4^yAIGMlewssvS*MNEPB7Ve}7wm@t0SN<=fR_`Rna(x6AuN{C2x~^A7a9e!Y0Vxc{AE>FwfvdAsJX8Loqn"
        "57#Sj+{gFVEAipu>cidoma8$}Q#N>m0Y~UzTTQncbPL2y<x$*#1Fh+;Mm|r6t|89w9g4&R#MvyCr<l$Gal#Wv7}t&<-<eX!PMgu*"
        "U-QzaH^1jS?S?}X{%sCVf?;vldo4G*p-cP)mSzDYsT)WYmI99&nxlf6(5>fVn{amqku>C^hfZx94J@rxmqomEFD8@XG)MYk>|ZdK"
        "z?*|cZ;1JkdRv@pqzjU;>42X@IJ%`jpDT>1le&Tfp6Izo)7E~lKrWW8+)i%_aC!JTA$_OddWCoV`2{#+7qDQyvOt={6Jxvt^il(G"
        ";}{P#$AP%zn|rMCsg~ol^>L$2CLJT{jQm7*-lZAXPUa35&tA+2z3}9X;n3Y0Q|)y#nMi(ynr-@C;_MMw!-%@qQa7XJLM5Ae>=FT@"
        "0(tJ+6ufM}IL>XtTPp_vgaL_!{&=<*gv5cptR6I7&({8X637N`i14X}mmjj>r@6Q;u~?uf0v_Y$*Qt<a3G!)&Tz{IqCzq7ZUhxIO"
        "TJDVD`6`W2r&s<o&jo^4D?%86KhjZlrQ(D1JYe;Rc%TSWZHJ$`WHUB4ZWGiOX4Ckw7IB*w2t-Je`CbQ(gt_C}G96(xHYM-O9{VTm"
        "LZi{hs+r_;MEcIAYx0YiNYVWb3{8mrV?@`4t)wxrp>$!B0eu=fW%1xA&e&?{no%3`pVOc0Qz!dWs)inimjI8h&J5?JZ3&tJ6_1w8"
        "vU{{P-O%~LG$1uJ_1Nh0e6~29N^cIHjbvv!pcRr~Bl+KFVVUvEs7;9F+vN>b-E&{;`8U#?)2o-SeuP}Gqxn3@jrg;8t!q=t=2z9W"
        "GS}4cN*@GO#*wFACl>i9h~ffYh@bae{z5VW`$V5IDEZ-P;A^_`%?AZ~oDZY|CQ`#^HZ(Pnli^JOMLDDn@k`%1z6YANT;E=S+J4d3"
        "kg=yKi~ppF<YeQO(+_8Gu$;f}B-i>MiB@1(Y`wKo6D7ihF_kT7L*aDYb_9R~yoXfpjz1p@{c@sa?(}G|)qem`O928D02BZK00;m8"
        "03iVBu1g=*0RRAq4FCWb00000000000001h0RR910Ap`%W@%@0FJ*XeWpZh4Wo~qHE^2dcZcs}B0u%rg000080000X06Nxj(_jPu"
        "0F4U(0384T00000000000Du7u0ssJGZ*FF3XLB!dWpHnEX>@OLd0%sMb!2%iYIARHP)h*<6aW+e000O8001EXT`4=JW&r>I?E(M*"
        "8UO$Q0000000000fB~ll003ieZf0p`b1!sqVQFqIWq4t2aBO8RYIARHP)h*<6aW+e000O8001EX=Zh95djS9di2?usDF6Tf00000"
        "00000fB{7a003opVQp}1WpgibWpHnEX>@OLd0%sMb!2%jWp-g~Uvh7Eb1rIgZ*FW*O928D02BZK00;m803iV7V;l{_0000R0RR9a"
        "00000000000001h0S5{I0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-AML|SOMJ{b*P)h*<6aW+e000O8001EXdo9;C-~a#siva)t"
        "X#fBK0000000000fB^~%003opVQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1yMsW@KSzHa0jkHaRjdV>M=GIAu07WivT4"
        "Gc`3ZVlg-{H8nIjIAt<8Wj10nWi~f9G+{DfG&DD3F*iA5H8d`2b8l`?O928D02BZK00;m803iUpmUo)o0001s0RRAL0000000000"
        "0001h0dWif0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFEM3eWHDqpV`MQhGh;C`Gc{#1VmUT7H8W*qG&L|WVl-qh"
        "Gc;ymWnwsCI5{(AW-&HlFlISnW@R@uFgGz{H7;s%Z*EXa0Rj{N6aWAK2mk;8ApqyD`R3pN004^t003zK0000000000004ji)(rpv"
        "Wq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3Z*+4nGh{R|VlrenF)(FiW;HWoV>n@9H8N#6WHV$mH8(V7HDWU`G-YHlVKgvh"
        "GdMFcH!)^qV`VvJF*r9kH8MFiE^2dcZcs}B0u%rg000080000X04^)lo#6lg0E+<t0BHaK00000000000Du8K4*&pVcwudDY-MvV"
        "a%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8>G+|_9GchtXWnwfjGB!A7F=I3`Vm2^2WHvA{WH4nkH92NEI59FdVm4%9Vlp@|G%z?g"
        "F=An2VKF&oGBh$SYIARHP)h*<6aW+e000O8001EX7KqqE-v9sriva)tX#fBK0000000000fB~ft003opVQp}1WpgibWpHnEX>@OL"
        "d0%sMb!2%jb8ca9b7*gLb1yYGHZWsiVK6l@Ic7CuH8C_XGc{&rW;J9mWHK~nHDWkpF*h_bI5%N9IW{?DFlAw5G&yEsIb}0sIWc57"
        "G&wG6b8l`?O928D02BZK00;m803iU(2MA%{0001r0RRAL00000000000001h0SgiU0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-h"
        "Zeeh9Xm50LFEuz~WMwpAVP<79GC43|GGt<8VlZQ5G&nIZHZo>8Gi5S3Vq-TpIWsk8V`eckGB##6G%+zZVlgu|V`XDFIWB5*Z*EXa"
        "0Rj{N6aWAK2mk;8ApkQw?GfMr004^t003zK0000000000004jiauWamWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3Z*+4n"
        "Ha0OZHDx$uWiw+rGBP<fVly!^H(_NsFfukVGdE>1W;rusGB;%~FgRp0GGsM5I5=ftW@I;FWi(?jH!x&3E^2dcZcs}B0u%rg00008"
        "0000X0CkQ;P~ZRn0E+<t0BHaK00000000000DuA66aWBacwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO9aIW{z8VL35m"
        "V_`L8H8VIdIWc8nG&eUiWMX4sW;ir5Ibt_qF*Y<cF)(5`F*#*nV=!bfG&MG2F*jvoVPa-3YIARHP)h*<6aW+e000O8001EX4ayOb"
        "-~a#siva)tX#fBK0000000000fB`@j003opVQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1!9NVmUc6H)1t4VPs@uG%_$T"
        "Vq`fnV>dE0I50UlHZ?eAWnwaBVK_B7Fkxk5V>L50W;bMFVKgx|VmCN7WMnRCb8l`?O928D02BZK00;m803iTY&)elg0ssIQ4*&ov"
        "00000000000001h0jC!L0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-ia$#w1Uvh7Eb1rIgZ*FW*O928D02BZK00;m803iSbyiLPq"
        "0001N0000S00000000000001h0Wul@0CRF<FJftCZ)0I>XJu_|VJ}}_X>MtBUtcb8c~DCM0u%rg000080000X0P`loJ(C3h0E`j<"
        "02}}S00000000000Du9~8UO%ua$_%IX=ZO@VQgn*ZEayMUteuuX>MO%E^v8JO928D02BZK00;m803iTbT`SE`000130000Y00000"
        "000000001h0iPfM0CRF<FJftCZ)0I>XJu_|VJ~54Wo~pYUtei%X>?y-E^v8JO928D02BZK00;m803iTk@JS?d0RRA*0ssIW00000"
        "000000001h0WKi`0CRF<FJftCZ)0I>XJu_|VJ~54Wo~pYY;SLHE^v8JO928D02BZK00;m803iScl}$BB0000=0000Z0000000000"
        "0001h0o@`10CRF<FJftCZ)0I>XJu_|VJ~5Gb7gdMFJE72ZfSI1UoLQYP)h*<6aW+e000O8001EXmZN1%;{pHxVF~~MCjbBd00000"
        "00000fB|MB0047xV=rQ9W^ZF*Y-eR{ZDB8Ab8}^Mb1!FQZEayRUu804E^2dcZcs}B0u%rg000080000X0JyAjh~okP04ECo04D$d"
        "00000000000Du9GCIA3)a$_%IX=ZO@VQgn*ZEayMVRLh3baO9fWo>O?G+$*jVlHZPZ*EXa0Rj{N6aWAK2mk;8ApnEz8iXAK001uv"
        "001ul0000000000004jiwkZGrb8=%ZVrgb?V_|G(Wo>O?FJW_YWpr~dbZ={AZfSaDa$jv>ZfRy^b963hb8l`?O928D02BZK00;m8"
        "03iU%U1_O~0RR950{{Ra00000000000001h0U#{^0CRF<FJftCZ)0I>XJu_|VJ~B7Wn*h_Z)t9HX>MmOaCuNm0Rj{N6aWAK2mk;8"
        "ApnQMrm1`d000dR000^Q0000000000004ji?JfWSb8=%ZVrgb?V_|G(Wo>O?FJo_RW@%?GaCuNm0Rj{N6aWAK2mk;8Apj!h{%0=("
        "0046d0012T0000000000004jip)>#hb8=%ZVrgb?V_|G(Wo>O?FJo_RbaG*1baO6nc~DCM0u%rg000080000X096G%QiA{h0I~o8"
        "03iSX00000000000Du7tH~;{1a$_%IX=ZO@VQgn*ZEayMWMOn+FJE72ZfSI1UoLQYP)h*<6aW+e000O8001EXXTKfly8-|JsRaN4"
        "ApigX0000000000fC0@o0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HVPs)&bY*gLE^v8JO928D02BZK00;m803iUT(M`e70RRAL0{{RV"
        "00000000000001h0mnN40CRF<FJftCZ)0I>XJu_|VJ~E1bYU-Hb963nc~DCM0u%rg000080000X0OQt%K@9@{0LKXc03iSX00000"
        "000000Du9}Jpcf6a$_%IX=ZO@VQgn*ZEayMWMOn+FJo_PY+-b7axQRrP)h*<6aW+e000O8001EXK1jz6$OZra5fT6ZAOHXW00000"
        "00000fB_vr0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HV{dMBWq5QhaCuNm0Rj{N6aWAK2mk;8Apm~(S)?`y006%n001Na0000000000"
        "004jiBS-)Mb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~E9VP|e{b98BAb1rasP)h*<6aW+e000O8001EX>|KCHPY3`2WE%hgBme*a00000"
        "00000fB~CO0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HZ)Rp}X>Mg-b7pidaCuNm0Rj{N6aWAK2mk;8AppTi@Igxl008zA001BW00000"
        "00000004jiELi{mb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~oDbYo~`b1rasP)h*<6aW+e000O8001EXJCxc_UJU>M?J58OBLDyZ00000"
        "00000fC05&0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_Ha%FIDb7^#Ma(OOrc~DCM0u%rg000080000X0RI=g%=ZWY0Hz!O03iSX00000"
        "000000Du8hZU6vta$_%IX=ZO@VQgn*ZEayMWMOn+FLQTgb8TsKXfAMhP)h*<6aW+e000O8001EX&twv_wFdwI)fE5$B>(^b00000"
        "00000fB}kk0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HbZ={AZfSa9bZKvHE^v8JO928D02BZK00;m803iUS1~;4O0002n0RR9e00000"
        "000000001h0fK)30CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFJE72ZfSI1UoLQYP)h*<6aW+e000O8001EXT*TlVVFv&J^cMgC"
        "C;$Ke0000000000fB~?80047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZBnaVQgPxZ)9aIaCuNm0Rj{N6aWAK2mk;8ApraYW(lqV"
        "008;|001Tc0000000000004jiSBL-rb8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XXkl`0Wpi^baCuNm0Rj{N6aWAK2mk;8Apqv;"
        "?GB_1003|$001li0000000000004jiNs0gfb8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XZeet3c4c31d30!RZZ2?nP)h*<6aW+e"
        "000O8001EXwiE8C*8u<k76bqQDgXcg0000000000fB`p_0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZB|hWn^h%bZKvHb1ras"
        "P)h*<6aW+e000O8001EX6y<DOKmY&$LjV8(CIA2c0000000000fB{yR0047xV=rQ9W^ZF*Y-eR{ZDB8IZf0e2Wo~0-FJE72ZfSI1"
        "UoLQYP)h*<6aW+e000O8001EXDv?LZ;0FKz<Qo70C;$Ke0000000000fC1W=0047xV=rQ9W^ZF*Y-eR{ZDB8IZf0e2Wo~0-FK1<L"
        "WpZJ3X>V>WaCuNm0Rj{N6aWAK2mk;8Apmvagc~UV0086y001KZ0000000000004ji0HFW?b8=%ZVrgb?V_|G(Wo>O?FKusRWo&aV"
        "Utei%X>?y-E^v8JO928D02BZK00;m803iT4l9bqU2LJ#*7XSbu00000000000001h0cxTE0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~"
        "FJf|OWM^eAaCuNm0Rj{N6aWAK2mk;8Api+!qYA$P002V+001BW0000000000004ji9I5~Sb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aV"
        "V_{=xWiD`eP)h*<6aW+e000O8001EX$VPJ(RRjP4r3(N6E&u=k0000000000fB_k-0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!3P"
        "Wn*h_Z)t9HUvzR|ZgXa3axQRrP)h*<6aW+e000O8001EX`6tF3a037Ug9!itF8}}l0000000000fC0L&0047xV=rQ9W^ZF*Y-eR{"
        "ZDB8MZ)9a`b1!3WWMyAxa$#_2UvP47YGq?|Z*nehc~DCM0u%rg000080000X0P5!>`^W+S03`<i03`qb00000000000Du8^v;Y8e"
        "a$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MVZDM6)WNB_^b1rasP)h*<6aW+e000O8001EXv;=<25(5AL0tf&ADF6Tf0000000000fB}NH"
        "0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!LbX>@5}Y-xI7bZKvHE^v8JO928D02BZK00;m803iUzq>>%S1^@ty761S%0000000000"
        "0001h0o%F&0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FKuOVXJuqxVRUq5ZggpHZZ2?nP)h*<6aW+e000O8001EX9+RSUu>t@9g9ZQq"
        "AOHXW0000000000fC1~l0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!XgWMynFaCuNm0Rj{N6aWAK2mk;8Apj!?hu{DY005*j001EX"
        "0000000000004ji+Qa|=b8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVZeet3c4aPbc~DCM0u%rg000080000X0IONywtoiz08JMF03`qb"
        "00000000000Du7*)Bpf;a$_%IX=ZO@VQgn*ZEayMZEs{{Y;!Mga%FUKVQFq<WG--dP)h*<6aW+e000O8001EX6-Rh_n*{&>BoF`q"
        "Bme*a0000000000fC1av0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!mbW@U0^ZewLGaCuNm0Rj{N6aWAK2mk;8ApnCIu8%zi006rX"
        "001HY0000000000004jixZwZ*b8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVa&L8XX>MmOaCuNm0Rj{N6aWAK2mk;8Apph#QCr>=002l;"
        "001Na0000000000004jiHRu2Ub8=%ZVrgb?V_|G(Wo>O?FLGsYZ*ysMZ*qBGb98lNc`k5yP)h*<6aW+e000O8001EXwfc{5-UR>v"
        "*$@B#8~^|S0000000000fB{wg0047xV=rQ9W^ZF*Y-eR{ZDB8SV_{}yZ){{PaCuNm0Rj{N6aWAK2mk;8ApqM@2_lXF006xJ001Qb"
        "0000000000004jiZ36)Sb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD?r0X>MtBUtcb8c~DCM0u%rg000080000X0EMl(<sJb50ObJy"
        "04V?f00000000000Du8L1OWhZa$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FJx(RbZKmCVRUJ4ZZ2?nP)h*<6aW+e000O8001EX-ZEcn"
        "V+sHO9wYz&DF6Tf0000000000fB~Zg0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPZDnn5a(Q2Ha&KZ~E^v8JO928D02BZK00;m8"
        "03iSxW4xPK0RRA$0ssIf00000000000001h0ZkAA0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rVVrpe$bZK^Fb1rasP)h*<6aW+e"
        "000O8001EXTJIlm`40d9W;*}?C;$Ke0000000000fC1_e0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPZ)Rp}X>Mg-b7pidaCuNm"
        "0Rj{N6aWAK2mk;8Apo>CALbJX003qj001KZ0000000000004jiDIx&?b8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@JXY-wY8E^v8J"
        "O928D02BZK00;m803iTufR+o10RRAr0ssIe00000000000001h0eva~0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rWa&Km7Y-w(1"
        "E^v8JO928D02BZK00;m803iVA%C(=s2mk;t9RL6;00000000000001h0ZJ?Z0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rYWpHnE"
        "X>@OLd0%s8bS`jtP)h*<6aW+e000O8001EX+W}19Bmw{c9|QmZBme*a0000000000fB{oA0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1"
        "X>MmPbaG*7Ze?;VaCuNm0Rj{N6aWAK2mk;8ApjsPk_V*@007%6000L70000000000004jixi<j-Oi4pUPE$otO9ci10000*07U>`"
        "7ytl+Mgaf-00"
    )
    bundle_sha = "a640b665636e173b0939edb45986a8e8e0357d6a516b53ab27350518b40599bc"
    bundle_bytes = base64.b85decode(bundle_payload)
    if hashlib.sha256(bundle_bytes).hexdigest() != bundle_sha:
        raise ValueError("Bundled source checksum mismatch.")
    runtime_root = (Path("outputs/repository_runtime") / bundle_sha[:16]).resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as _archive:
        for _entry in _archive.infolist():
            _destination = (runtime_root / _entry.filename).resolve()
            if not _destination.is_relative_to(runtime_root):
                raise ValueError("Bundled source path escape.")
            _destination.parent.mkdir(parents=True, exist_ok=True)
            _destination.write_bytes(_archive.read(_entry))
    if "bifocalgemma" in sys.modules:
        _loaded_root = Path(sys.modules["bifocalgemma"].__file__).resolve().parent
        if not _loaded_root.is_relative_to(runtime_root):
            for _module_name, _module in list(sys.modules.items()):
                if _module_name == "bifocalgemma" or _module_name.startswith("bifocalgemma."):
                    _file = getattr(_module, "__file__", None)
                    if _file and Path(_file).suffix == ".py":
                        _path = Path(_file).resolve()
                        if not _path.is_relative_to(_loaded_root):
                            raise RuntimeError(
                                "Restart runtime: mixed BifocalGemma source modules."
                            )
                        _expected = (
                            runtime_root / "src/bifocalgemma" / _path.relative_to(_loaded_root)
                        )
                        if not _expected.exists() or _expected.read_bytes() != _path.read_bytes():
                            raise RuntimeError(
                                "Restart runtime to switch BifocalGemma source versions."
                            )
    sys.path.insert(0, str(runtime_root / "src"))
    from bifocalgemma.config import get_experiment
    from bifocalgemma.data.repository import SOURCE_REVISIONS
    from bifocalgemma.models.native import build_native_model
    from bifocalgemma.models.pretrained import load_local_native_model
    from bifocalgemma.repository_study import (
        fetch_tokenizer_assets,
        generate_run,
        prepare_run,
        read_source_rows,
        score_run,
        write_json,
    )
    from bifocalgemma.training.memory_probe import tensor_bytes
    from bifocalgemma.training.repository_sft import fit_repository_sft, prepare_repository_training

    environment = {
        "torch": str(torch.__version__),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_sha256": bundle_sha,
    }
    mo.vstack([mo.md("## 2 · Runtime/source"), mo.ui.table([environment], selection=None)])
    return (
        Path,
        SimpleNamespace,
        torch,
        HfApi,
        hf_hub_download,
        os,
        sys,
        json,
        runtime_root,
        get_experiment,
        fetch_tokenizer_assets,
        prepare_run,
        generate_run,
        score_run,
        read_source_rows,
        write_json,
        build_native_model,
        load_local_native_model,
        tensor_bytes,
        prepare_repository_training,
        fit_repository_sft,
        SOURCE_REVISIONS,
    )


@app.cell
def _(mo):
    hf_form = mo.ui.text(kind="password", label="HF read token (optional)").form(
        submit_button_label="Apply HF token"
    )
    hf_form
    return (hf_form,)


@app.cell
def _(HfApi, hf_form, mo, os):
    hf_token = hf_form.value or os.environ.get("HF_TOKEN") or False
    auth_status = "anonymous"
    if hf_token:
        HfApi(token=hf_token).whoami()
        auth_status = "verified read/download credential"
    mo.md("## 4 · HF authentication\n" + auth_status)
    return (hf_token,)


@app.cell
def _(
    SOURCE,
    ROWS_FILE,
    PREPARED_DIR,
    TOKENIZER_CACHE,
    RUN_PREPARE,
    FETCH_SOURCE_ROWS,
    FETCH_TOKENIZERS,
    ALLOW_GIT_NETWORK,
    CONTEXT_LIMIT,
    RAW_LIMIT,
    TASK_LIMIT,
    PER_REPO_LIMIT,
    SOURCE_REVISIONS,
    Path,
    SimpleNamespace,
    runtime_root,
    get_experiment,
    fetch_tokenizer_assets,
    hf_hub_download,
    hf_token,
    prepare_run,
    json,
    mo,
):
    preparation_result = {"status": "DISABLED"}
    if RUN_PREPARE:
        _spec = get_experiment("BifocalGemma-E2B-E2B")
        if FETCH_TOKENIZERS:
            fetch_tokenizer_assets(_spec, Path(TOKENIZER_CACHE), token=hf_token)
            fetch_tokenizer_assets(
                get_experiment("BifocalGemma-E4B-E4B"), Path(TOKENIZER_CACHE), token=hf_token
            )
        if not SOURCE.startswith("fixture") and not ROWS_FILE and not FETCH_SOURCE_ROWS:
            raise ValueError(
                "Real sources require ROWS_FILE or explicit pinned fetch; fixtures are never relabeled as real data."
            )
        _snapshots = (
            runtime_root / "examples/repository_study/snapshots"
            if SOURCE.startswith("fixture")
            else None
        )
        _rows = (
            Path(ROWS_FILE)
            if ROWS_FILE
            else runtime_root
            / "examples/repository_study"
            / ("eval_rows.jsonl" if SOURCE == "fixture_eval" else "train_rows.jsonl")
        )
        if FETCH_SOURCE_ROWS:
            _files = {
                "swe_fixer": "swe-fixer-train-110k.jsonl",
                "swe_rebench_v2": "data/train-00000-of-00001.parquet",
                "resq": "dataset.json",
            }
            if SOURCE not in _files:
                raise ValueError("Source-row fetching is only for the three pinned real sources.")
            _ref = SOURCE_REVISIONS[SOURCE]
            _rows = Path(
                hf_hub_download(
                    _ref[0], _files[SOURCE], repo_type="dataset", revision=_ref[1], token=hf_token
                )
            )
        if not _rows.is_file():
            raise FileNotFoundError("Select source rows or enable explicit pinned source fetch.")
        _args = SimpleNamespace(
            source=SOURCE,
            rows=_rows,
            snapshots=_snapshots,
            git_cache=Path("outputs/git_snapshots"),
            tokenizer_cache=Path(TOKENIZER_CACHE),
            fetch_tokenizers=False,
            allow_network=ALLOW_GIT_NETWORK,
            study_config=runtime_root / "configs/repository_study.json",
            output=Path(PREPARED_DIR),
            context_limit=CONTEXT_LIMIT,
            raw_limit=RAW_LIMIT,
            limit=TASK_LIMIT,
            per_repo_limit=PER_REPO_LIMIT,
        )
        prepare_run(_args)
        preparation_result = json.loads(
            (Path(PREPARED_DIR) / "preparation_report.json").read_text(encoding="utf-8")
        )
    mo.vstack([mo.md("## 5 · Preparation"), mo.ui.table([preparation_result], selection=None)])
    return (preparation_result,)


@app.cell
def _(
    RUN_TRAINING,
    PREPARED_DIR,
    EXPERIMENT,
    TOKENIZER_CACHE,
    CONTEXT_LIMIT,
    TRAIN_DECODER_LIMIT,
    Path,
    get_experiment,
    prepare_repository_training,
    preparation_result,
    mo,
):
    train_data = dev_data = training_manifest = None
    if RUN_TRAINING:
        train_data, dev_data, training_manifest = prepare_repository_training(
            Path(PREPARED_DIR) / "train.jsonl",
            get_experiment(EXPERIMENT),
            Path(TOKENIZER_CACHE),
            encoder_limit=CONTEXT_LIMIT,
            decoder_limit=TRAIN_DECODER_LIMIT,
        )
    mo.md(
        "## 6 · Matched cohort/tokenization\n"
        + (
            "Train/dev prepared with repo grouping and all-six-model eligibility."
            if training_manifest
            else "Training disabled."
        )
    )
    return (train_data, dev_data, training_manifest)


@app.cell
def _(
    RUN_TRAINING,
    RUN_GENERATE,
    EXPERIMENT,
    MODEL_CACHE,
    HEADROOM_GIB,
    torch,
    Path,
    get_experiment,
    build_native_model,
    tensor_bytes,
    hf_hub_download,
    hf_token,
    mo,
):
    model_paths = None
    model_guard = "Actions disabled; no weights downloaded."
    if RUN_TRAINING or RUN_GENERATE:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            model_guard = "SKIPPED: supported BF16 CUDA is required."
        else:
            _spec = get_experiment(EXPERIMENT)
            _blueprint = build_native_model(_spec, tiny=False, device="meta", dtype=torch.bfloat16)
            _weights = tensor_bytes(_blueprint.parameters())
            del _blueprint
            _free, _total = torch.cuda.mem_get_info()
            if _free < _weights + int(HEADROOM_GIB * 2**30):
                model_guard = "SKIPPED_HEADROOM: free VRAM below weights + configured reserve."
            else:
                _dec = Path(
                    hf_hub_download(
                        _spec.decoder.model_id,
                        "model.safetensors",
                        revision=_spec.decoder.revision,
                        cache_dir=MODEL_CACHE,
                        token=hf_token,
                    )
                )
                _enc = (
                    Path(
                        hf_hub_download(
                            _spec.encoder.model_id,
                            "model.safetensors",
                            revision=_spec.encoder.revision,
                            cache_dir=MODEL_CACHE,
                            token=hf_token,
                        )
                    )
                    if _spec.encoder
                    else None
                )
                model_paths = (_dec, _enc)
                model_guard = (
                    "Pinned sources cached. Guard is not a guarantee of full-shape training."
                )
    mo.md("## 7 · GPU preflight/checkpoint cache\n" + model_guard)
    return (model_paths, model_guard)


@app.cell
def _(
    RUN_TRAINING,
    RUN_DIR,
    EXPERIMENT,
    MAX_STEPS,
    EPOCHS,
    ACCUMULATION,
    model_paths,
    model_guard,
    training_manifest,
    train_data,
    dev_data,
    Path,
    get_experiment,
    load_local_native_model,
    fit_repository_sft,
    write_json,
    torch,
    mo,
):
    training_result = {"status": "DISABLED", "reason": model_guard}
    if RUN_TRAINING and model_paths:
        if Path(RUN_DIR).exists() and any(Path(RUN_DIR).iterdir()):
            raise FileExistsError(
                "Choose a fresh RUN_DIR; this notebook does not silently resume/overwrite."
            )
        if training_manifest is None:
            raise ValueError("Prepare matched train/dev data first.")
        _model, _provenance = load_local_native_model(
            get_experiment(EXPERIMENT), model_paths[0], model_paths[1]
        )
        try:
            training_result = fit_repository_sft(
                _model,
                train_data,
                dev_data,
                Path(RUN_DIR),
                training_manifest,
                max_steps=MAX_STEPS,
                epochs=EPOCHS,
                accumulation=ACCUMULATION,
            )
            write_json(Path(RUN_DIR) / "data_manifest.json", training_manifest)
            write_json(Path(RUN_DIR) / "training_report.json", training_result)
            write_json(Path(RUN_DIR) / "model_provenance.json", _provenance)
        finally:
            del _model
            torch.cuda.empty_cache()
    mo.vstack([mo.md("## 8 · SFT (one model)"), mo.ui.table([training_result], selection=None)])
    return (training_result,)


@app.cell
def _(
    RUN_GENERATE,
    EXPERIMENT,
    PREPARED_DIR,
    PREDICTIONS_FILE,
    ADAPTER_CHECKPOINT,
    TOKENIZER_CACHE,
    CONTEXT_LIMIT,
    GENERATION_DECODER_LIMIT,
    MAX_NEW_TOKENS,
    TASK_LIMIT,
    model_paths,
    model_guard,
    Path,
    SimpleNamespace,
    generate_run,
    mo,
):
    generation_result = {"status": "DISABLED", "reason": model_guard}
    if RUN_GENERATE and model_paths:
        _args = SimpleNamespace(
            prepared=Path(PREPARED_DIR),
            experiment=EXPERIMENT,
            decoder_checkpoint=model_paths[0],
            encoder_checkpoint=model_paths[1],
            adapter_checkpoint=Path(ADAPTER_CHECKPOINT) if ADAPTER_CHECKPOINT else None,
            tokenizer_cache=Path(TOKENIZER_CACHE),
            context_limit=CONTEXT_LIMIT,
            decoder_limit=GENERATION_DECODER_LIMIT,
            max_new_tokens=MAX_NEW_TOKENS,
            limit=TASK_LIMIT,
            output=Path(PREDICTIONS_FILE),
        )
        generate_run(_args)
        generation_result = {
            "status": "COMPLETED",
            "predictions": PREDICTIONS_FILE,
            "backend": "eager/no temporal cache",
        }
    mo.vstack([mo.md("## 9 · Generation"), mo.ui.table([generation_result], selection=None)])
    return (generation_result,)


@app.cell
def _(
    RUN_SCORE,
    PREPARED_DIR,
    PREDICTIONS_FILE,
    SCORE_FILE,
    PYTHON_EVAL,
    VALIDATE_GOLD,
    ALLOW_CODE_EXECUTION,
    Path,
    SimpleNamespace,
    sys,
    score_run,
    json,
    generation_result,
    mo,
):
    score_result = {"status": "DISABLED"}
    if RUN_SCORE:
        _args = SimpleNamespace(
            prepared=Path(PREPARED_DIR),
            predictions=None if VALIDATE_GOLD else Path(PREDICTIONS_FILE),
            output=Path(SCORE_FILE),
            python_eval=Path(PYTHON_EVAL or sys.executable),
            allow_code_execution=ALLOW_CODE_EXECUTION,
            validate_gold=VALIDATE_GOLD,
            timeout=30,
        )
        score_run(_args)
        score_result = json.loads(Path(SCORE_FILE).read_text(encoding="utf-8"))
    mo.vstack(
        [
            mo.md("## 10 · Native CPU score / gold validation"),
            mo.ui.table(
                [{key: value for key, value in score_result.items() if key != "results"}],
                selection=None,
            ),
        ]
    )
    return (score_result,)


@app.cell
def _(preparation_result, training_result, generation_result, score_result, mo):
    mo.vstack(
        [
            mo.md(
                "## 11 · Run summary\nPreparation validates static patches; gold validation is not a model benchmark score. Empty/missing predictions and infrastructure errors remain visible."
            ),
            mo.ui.table(
                [
                    {
                        "stage": name,
                        "status": result.get("status", result.get("purpose", "prepared")),
                    }
                    for name, result in (
                        ("prepare", preparation_result),
                        ("train", training_result),
                        ("generate", generation_result),
                        ("score", score_result),
                    )
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(SOURCE_REVISIONS):
    # This cell defines a pytest test; notebook actions remain disabled by default.
    def test_actions_disabled_by_default():
        assert SOURCE_REVISIONS["resq"][2] == "eval"
        assert SOURCE_REVISIONS["fixture"][2] == "train"

    return (test_actions_disabled_by_default,)


if __name__ == "__main__":
    app.run()
