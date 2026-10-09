# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo==0.25.1", "torch==2.14.1", "transformers==5.19.0", "safetensors==0.8.0", "pyarrow==25.0.1", "pytest==8.4.2", "huggingface-hub==1.33.0", "tokenizers==0.23.2", "numpy==2.5.3"]
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
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
    CONTEXT_LIMIT = 2048  # bounded pilot; choose a new pool for longer contexts
    TRAIN_DECODER_LIMIT = 2048
    MAX_NEW_TOKENS = 512
    GENERATION_DECODER_LIMIT = 8192
    RAW_LIMIT = 1000
    TASK_LIMIT = 32
    PER_REPO_LIMIT = 8
    MAX_STEPS = 0  # 0 = complete configured epochs
    EPOCHS = 1
    RESUME_FROM = ""  # checkpoint-v2 path, empty starts a new run
    TRAINING_DATA_MODE = "curated_pool"  # curated_pool / repository_tasks
    CURATED_POOL_DIR = "outputs/full_repository_pool_v1"
    FOUNDATION_POOL_DIR = "outputs/full_training_pool_v1"
    PREPARE_FULL_POOL = False
    CORPUS_KIND = "repository"
    REPOSITORY_SOURCE = "swe_rebench_v2"
    REPOSITORY_TARGET_COUNT = 10000
    REPOSITORY_PER_REPO_LIMIT = 50
    REPOSITORY_DISK_GIB = 20
    ACCUMULATION = 4
    HEADROOM_GIB = 16
    return (
        RESUME_FROM,
        TRAINING_DATA_MODE,
        CURATED_POOL_DIR,
        FOUNDATION_POOL_DIR,
        PREPARE_FULL_POOL,
        CORPUS_KIND,
        REPOSITORY_SOURCE,
        REPOSITORY_TARGET_COUNT,
        REPOSITORY_PER_REPO_LIMIT,
        REPOSITORY_DISK_GIB,
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
        "0NxMFF|-%}05Vem03iSX0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-Jb#h^JX>V>WaCx;nYj@i=mfw3${{xoU-Juehk?h!YoYCZ@iQ4f_"
        "YG-VBI&&NkOOc?(nj%?}itTpt-}gQM5+otnnX{`tECLt6#l?N&QjX)iSw~47EK(`rb(-#kd0DJ0A)ztJu0;^7lB!Db?3KvmCs_)4"
        "C&P6uD^ca^GL$0BBk4Jg^YvF>f0e9?ysSk~)y?;1P%YDB(L5)4^Yxc1&zgr)Hec)fMrO%x&=8mT3KmFH8P<S8B^1Qle4W*@9E(WC"
        "!8)y@B&-#55!ASzhWr4Jh#B=Qk~j~8^jfY~ffogJ;N@|gCYkiBxYi@o#{E(PC8DvYh&Fx<ZL%!NB&#>hR>~r;k~%N%MYF3S_mfCw"
        "byD9W9_Ob&p1qy>XY=pQ#Z3JE^;ZHu4z5`Q;Z0mSN4k+Zk_Z}{Y+aP;^Em*00HE(KoUs8?%1;TN4+2gOzFE8o4-a?a7cXQOOkeKp"
        "zKmrQ?S8ZWV(%~x_rl4+B;JcV6Cmo9to<|zWmZX?XO+|%<g@zgg!yWvvGj}kdPz$TkzFL$$P&*@_>Bn1;r>A^_W-ZGNq7(i!HemO"
        "aCb4C?uLN-;!w(Ha=0%K{vCufc~>JQr4`Lp5|(+D$My3QfP0H@?|9yg$iCctA$RxUU^forZnPUM_74ujz4$Pm9!{qFi-YMwu)p`s"
        "bpH#H{rvS;ALeI29A8|_-}!G(-_8B^$8YB+=W~-oj4@6CL|w|AU2oDfEvjf|cPC7PH7sE^?)4j!Dr~-o;bdoO%(BijmW&qu0}@6e"
        "?vqqjE`A>!sR532e4XVbXpX#G1vSbIojdS*H3sIdgCrAa0-_qEl?bv(uq?z0D3Yp>DG1*+D-WD6p~<3l+xXGiz?+`{agI;EA6xt0"
        "{qW}OFY_~7=iAdCewd$J*t$;MTU+Npy+1yg+q#ZV-ffub-Q4K;%zx_g{-^~7daTP#)axRZZre<NUR8r?x$d?CRR!p!tX<?<W<blj"
        "IgKRu&S~KBDW44Jpddn(HtJ55#IQzvUr3SVG^L_-3#A6gyHFOjaGNDB@XNd`b5OU(86IDNx>&WG-^24AJ&ZPOwUi!c&nQS!w{$+<"
        "KKs|@KR@1ntgf6fan@+m<FpEJ3t+tLFq@iksE*wMdk!XC0(2_5cPws`s9w$v^~5|9m~Jo+_dgNmOVs7a*m+i@0T{iw46cdGMNl$H"
        "WtOk6moOPvze)o5LT2G|6_ht32eX+5g{PQ27OBjbzST(lQS9vw2?qw|Hx$I$9d&l!wzor&1j@>K)iT)KKX9Fo6UX}{PxLaEi8vC8"
        "c*a}58hL;lY|tHzyrsO0l51JPN|#5|{j0H<9vHlXESMm_lmQUjH8J1e*<jYG8ldP78mFoRs?<~7IkL@Nt25?zK?vkr%J7Dj0?oLi"
        "PD%Vv5QYI--n>9Y1Gc#r3os*~7#8>9{IBmppLn<qPWNp?<zLsbOu#JxWhc;4xfg0ePyCeB@O7ui^Awzzpai!eS{EV=GJ+0xs#TID"
        "tMzIJX^DMN0=ip@rCgUuRVU#qQOQ*R+AD<9xGfX(fl6or;^gbP0?j6>FfXLnY*ZtOjOhC?eeL_k!BsMi$KvT&)XP#<%RG%{6Yp@`"
        "uH7*1>?F@5x*|F7o%GSiQB5fBex9ZGv+qE~N_7rL-9@}+0pRlHdk<!~DvHFQu{TQ6S%Sj;-0LtV8T2I?oKH=P-f-YtVCzv2GFku("
        "?$luSxI+h#R3v{0bkx&%?1NRmq6OO1hoL_-ct#GQYfd;Cf_Vi<c~*O?n<yz=eyC;_<r)q^-T`y?`3*e|m2PuBJ>0rGR4963o@L<b"
        "y0j=9Q=WbTA@zdF2Oe}sAZpI*70A`B%t3iVn{8G6kmVqs9=OwW=#*fDgHm7N7jRV3T361<f|w>V<L4=l-*wLC@8@qX#M{#!PcGc2"
        "Bk|qY=?}sItZ1o`7t1;Xw-8XioLqrsn{JpJKV^`B=>087*OIJ<>l}klT7kU)nQL`tf~v!FEL&ua*lpiV(cUCa#KjVr2mTF2lvF?w"
        "MKD-#tb&~Y@VphuhZ8{5aE<0Sf@4SC@~q2if>mLe6tz1<-FbI*`axX0`S$%>9DgU~e?LCII4AQ{y%N?weOw#5>3m;_w`cP=7jxB1"
        ")44kK5Et{mUx*K9$KWRYL;Pj_kFmHegH<KIeSiAxs5g|Ra~m^k8QYn*KU^d0;tJyU<YN9kIA)z8V`DY~Yr01x@ze3epHF|h5ND@9"
        "9lv|!bm=!Tm1+=0E^AReT)7km<7q8Wmun0X4F%KeZ)Wv7T1ckU%;HQKK*Tqs)2bYm8*Qrsqq<4GUNtgiQN0qBIj`H*e2T$0E4_ix"
        "9z!H?b|FsB#PRnhr)P6KkW)Tl@wYece}pi@eLa3X0<F*<c}vhyPihuvPVz<M^f0`0oj;-uaQ^Cur{vo3#=)|62r+szo@et9@87&t"
        "h!PtQ5N(ee@p)iwFxvMvDYR2Yb8m0W;dqxBx1|(mUE+E4ubCZ2;{p5XuXx0^!=t0|K;J0aXgw9nIL$kk8t4knsmDV2v0%B$IB7$D"
        "iSSn*ikpE!D~AyKg-a!g{}1x_D8?`ITja|gTUm}>S+PbRkriWk`03AcV1ova+3T)Q^r&tcyl#*7iU23Hgy;tVb|lkXXU4(<daS_<"
        "*V15uE?pRU3spF7_}#gIqo!OEM(Qa9c9GwB1*ME7_%tDAC#VtHs$`X<0eHFh&^YxbpTATc#)J~V+HvYqqS1jkbMM&x&lGA3zM7h5"
        "5cm=jm{&vuRD#j{QYJ*%%HtTFg(ha2j7clD8wAyOIGNZSuW-4}ZcMkkx>U0r>1m(g*sI}Cic8V0mnCk%8LI%$d9BqlP2N|PbRPL_"
        "^(fK7PhWh-U2m|_o-9hr0{|D)KHIYn-kV^Yk@)`X^v4h4+kZ6c+B9-QLQCHUC)xBH;K=FJfE*hfSLP2x!9JUD+g6F@QQo@16Sa`F"
        "(L|<ZHsry?6T3R9nUn(?BDa6I{W!kat4}L1S}Qm)pK}{-pC=#1^Z3zT!56U*>k(obw(`s~8co)|4bqka>_r?yfV^IU6OVX-0%eb7"
        "oeg{z%4(1wA)Y@Mha<aFVL90D5>^x3qbDkriZ3%@Oyewx|K@r6X1&qD0Nfe-oRMZ#1{=W#o<Zz%AcqmDG3lRXz4U-(f7*a-I7)3+"
        "D21c7t^W(Ba5l+k$KW&vyaBdjh}s7WxM89?CZ_dM!ZfcW`6(R_<xBi(zSig){HB){uy;0@44s>7E+~9ZIW7d6lni1$ukkBNh>2^|"
        "P81q5-vB6yy~Di@aBS1m0_vbyS>mKhvZ@Y%=&qg=L}G+NZLhnXf<esm<REHqH$nG=*|pD^?5ekXDLwWQOXrNUX9D<DTqmDoCiprD"
        "H;WwoaRGv_ag5X8x|+f0;|K7_Gkq(i7LJj@kXjO)>7b;gfByMr_w^CO{j0bieXO2-%v|s3>(R#y`rz?DZEK&md*NAuPw*^}cns{3"
        "Hg$0UuvY&u6T6S3TCiFq*Xw*;`AJr+YoF%VTUf|AC;ymQz3Vbx7w(kuKp!(@QuB2U<9MHVfvc$ud+G1?^>Dk6$y)Xus|D1Sjcq*<"
        "Z_s>$DNz^4kpv{FBF`$KP>`nit&GNEmE3X4Bn>jUJa|s~X_Dt&M;3{OY@jq^eFySU^klrd1jSDf*QBxA>AA0`;58UeW2Zfe3|bWy"
        "OXdXdPx-V!no3X<S2j@O$@RjpkEQd^hr~hzl_&ru?2mJ#E#A)?^M3T`K-)<j^T7Z#T3hiNH#z)Aq5D&VE2{63>i{U5pXJIZ)^I3v"
        "d$nx&f^b9`1Ki9-)Z%mvR%ivPGIXlYU>Q_2cbQx#S&$0jwjC%0^g`w3i4#iPp?d`GTtHAF3syQUKeT5K7(PEvQo5V1@^p>(ctyv<"
        "1O~f(5C-Q7b1=T(RskFfZ+{X4fH=(TH|T+e(RyI&9v=F8F*7e`<{KRQoilY2@vx)Qx-MWz8LZGCIXdKMuQZ!WHA;f|MF8g>`FuQ;"
        ";tusR)(T7|a(zy#a_dvwo91|};}bHf4Pw)v(l;JMFuzLIz_6~~C&`AaI`DYK!|NG0C0~DK+*+T>AVT#-zWytYd7rnDG1Y*mOCsdx"
        "3a>mv@qxwh*EK+a_jHZUd3$*cE{`nuQseSU^V%=qg>{;6D)UaHlHfKWA<ahG0OG)*X!jn?kV5m`=;glk;%Iu%yEf_(rz7-=(1=N="
        "aK*hF>vQ2G*By#==kyfj*vT9)=CBgO73PSZ!^JYKE~AaAm<LrokMv=rkF`lP(D#TgA-Lm_(R>@IlkbE`$ZWl$%XJqq?}tJv^m72%"
        "p}23_?eAXo>yN=xGn>E1D9n`4XAovKGYeQadjTV?!8=UdEn3<TceTij)iN`)^2k^n{1<Dz597f=j|#E1^OxUVI>>I$l_#^Px=re("
        ">!AMJbp5o=!6U}CT?r|D=cBq;*|$aqUTM?fp;J=!eRPFvX9mTb8;WxtjX3G)W@3|Vk!#Hrt+!ktT&O=btkxA%y~kEu?ZX-wM#El!"
        "HUoe0a{LM_GKgqw9j-Qv20Gn!4Ym2PgyCJrGT55k1um1@lyeypR0kTZ1K_s~fQ>IGz=J&^{0L%gt>=IW_NxI+TCC{9{FLiFN0(&>"
        "T>GBlV|zpdB~OEB3d>le<I($Tji#|fRYYRj9xhJa7vA;K`LQqHT&d$EM+1&?^HnbvjIhfc*^(Taam~Q<_ZRoImF{YC!fiI|;Id^A"
        "yCyY1L4`L@Wi_}c=znxElToEKHqal7GYP&-rOTN{=weI!_5cDyI)J|kA{Y7K^FtV}6;(y+0@qN%D9vo2Se-5ide380LGzY<VuZ${"
        "zvSSy@6TXC(q4K2R9Nc!C8vI_Fdpi*x5*ugD$pOxX3z_wo5ewamwg(y=Kyy&e>Cp<#~n$qB@?ZBgOH+}tM=0#EymzIf^tW1%)D~}"
        "drRbMC%nY}+jK2h;6;*=>(JSo3PItIxhISBfu&9}!<%fFiRay5uL1n=Mj7XBH`?qD+dG_KS<AqS?(x||Xm$eo;wrajEVf|Wrz73J"
        "Z<@*KK5b6S<7Y$S6F<rog+R&hUoW}w>JsaWuQnt#jnV#%q{JpW%@2bc1<BS`oBpk9537=`ll8&2&T0BNFrn$#Y#UbQ997rAUD>9L"
        "F+pyNPVE>4;4}hfroIk$%}CQagk`K%8^buer)IXyWr#hprjLb%Fdu&3HcZ8rKE{`nZlLJ1^?gg_M#k$b<N`^bY!7ADHWBdiqZA(P"
        "P3M7qWZ*AS$TVCDh4wxAYzys3fX69d9aDQjQ9#^)3t06ehh`eu8lNdwpjBSmY&~#UMTsxe9R##xSsmq*vf+`+sRsRn_fj@=coiO~"
        "clUd%P;?Bvf#)%QgX<7S<Co9Fdmfz`At$>Vz5bWjpG+p`l1(g|*8{Ei@I=r;6`TAtGFPVrht?oi;Az11vm4#(5|x?Lr#SB9=oA|@"
        "%hNL5gccK%NWx#cske@yDsuROdY4NhML+k$8JQwoSK`(v9#*Bv;!l)t5$^P$TQ#gpm@U=AqgQN?2W8<{f?U~4plO{$qwd(ayTh<4"
        "lqJ1O(Ok3{IXdD%UzI;nu!5O^%2j{qstIgoG@E{duTg#S<1!$10!vOisj?y`0}L@Mu@bAk9ULo~=_#3>XH7*sxP}P|JSYz1^$t!3"
        "OvE2M3-$^{H~ZG+ARTUSF)rSvVrfTPq^`C(8g=TF@HGTCG2@`yWdii(SuT?+3{aDvOthUd{k>`G-|w~{iP5DvJJG11&tjze=x(0W"
        "WiJh-_F&)x&H`lvlmR+woWOzIGEt*Z&u|#pKc%N2SnWiuI7{DPnv*;cZ<z^@)A9AcWq|MM1o#>i1;97ebCv27t94cD+W0z$fM_X8"
        "&xQ~4Rl5z9+&0%mBw+yJh~tJ83aS%{YSU(X9O~rp4CyOEU&6?~W4hs%zX8XKc<wehzFWqQoH+X8lWnJKa2;Ab_3lUui|>zYziEUc"
        "MQ33Z+(<u4Dy#{reNqluga8v+nuV)2q`Xf!XXTm>571+l2CGFBaLtU(Q$Rvfx{+os``!g{pf0)rVeMW*aWD;tt%5suiga89IZW{u"
        "Pdlna$92$V<lM=#G@fZum7G?aH%J&stY8Y%%@@w3s19F|+zKgFg-t6LV9*`_qJto1ObG+)5}v%P)frdqA)}gPtkvy2tg`i#<*ERW"
        "vDNmS1E!=rHncf;*VX<uuLCg_1g!FtL$+XKGNz)|hRP_;?1F7nOOnHg-#RwkytBaN#wIO38Be`v%W3U2CEHj%_*&TkYZ-fPnYHf("
        "J4V-0UxNCuOcw@IwCO=0osOEr@Ao2B=%C1s1EL2utg13a=t&u^#)!|YENGifFOvjt;<xAyR{T<fIuKQ6TijDW-4zGg-k=KukKRhI"
        "FqOCp3ap0W;pj^`l=x!hePkY0KM{ur(4{y`mK33v7g+Q|4yrr2eo-DbfWASLj@gR$%W7zp2h_T8C!<HKhL^CzwLoUTECn^}ibsv{"
        "t?8BWdi(|EJ>>9Zm)_fLJk%Qwv}I236PZSw*i+*$@kawdd3$vZ5gR!r+e^L?1scSuW1MLFVg%fsUh@ai##e9inY_VGK*-)}f+K@H"
        "=shSJM09qYa=~n|1WlL9o<iV4RV*OKdo?x!M+%f)L3O8--4~#p@edZ%IRn}=Qabs<#i<>op#Y2*6x*G0xfw(%#148Gd$#<<Z#e2e"
        "JCf$p(=Lpkf?HkBb|)_myWqo)MNmol!2s@gaWK(6wu|T>!fQE-(A83wyXX(R3H8q6I!k_COH#P1_$Mx<{0%ETu-=HT+@EVSz(C`x"
        "sjZ7W{9yvnk8yIX)x-SnALeJr_%^hE{$c+1eA6gWeE4p*aRXueMuL7h*u0=%@@tjTMsXaCRa69hT`$Rfy&9|k{tY)Nqq34gjd<)l"
        "dGbUY8`Ut`?`WL<V3bk(;p+28Xh*>BfaFJlA&~2-iyxWp&|uUoo2sUkD|FtUJ_TrshzTP~6wx;$)iTteD#^W>fVTQzSkaQ!>7V80"
        "#Fn1x%7OXm%7kT%V)E(Ubn?ya<_Wfe^<N&+eEaWEV7TEIz(f2Rtlo<qbt|S@5v>5Ts{1<x`s1i2lna0d>giRJXy8xK5K8|kB+PLd"
        "ucDN3a!13JFG=0%GKrdZ#MCx(<3-p8*#<Nktc#9r-hxsCvK_5nrZI^vYVd(-@*oAPl41IUYov!0=oi?P$c=Fg@IYml01UsN&acMW"
        "ZVd2jFZD6{^WAURYwa}K?o;{$U+};N+lNLFG=?($=5aem4_~D-btFgwhp9e`#0*ycQ&%*GBQQwdK^<<zksa~bHgdpRIJ#>ydtzrz"
        "zLLSvdmaMdXWO-m8Dbt;a@t#OBXnL>RkUN0E>82?trrvOl&C)aH@WG4;o%hghU5e7fYH$_H4JMJo~=3aN2kGir||p3R5gg#rp-Th"
        "o%Bvp+D+)UBetZYja0Pq^>A$(2Fo@1epX@nv03xiHD*n_zd^vUJkqe|(Z=Lq<ImM_H(1jiGMoWv(<FD7H0G+UrcpzCr`n*Q;jLyR"
        "4HhyzQj1(t>I65%&!lHD+0i6AEb0l-pjpNZm<sia{Rss26K|iw{Nm+4J-yVAFZuB$J%WCM$7x@2e(t#D_8h%&L}m)aOv7(FvaOl?"
        "PM*J^rp^M8Rsq<iqgw$BomKY-ps1rY;vKcvkJx*LyQ2vEuolbORehG=o(5LyLYAMBN=C|bp#-g0m@Vr`{iU||M&_(fL*ovBf26?2"
        "^h}%g=X&-rC4S51J(DueYyE2&3^8guw*T-2{qTrRHTA@;Nj>rJQ0SZ!rBF=WuYQb!uy`(99`Z~$9{xsLYStF1Tt>ZAVJc{5K0=6A"
        "X3n~fcMcu=^@dxd`wZ(O>`|=7PN%k*JQnTCFYoq~Tq?`8p-|$9IOE^GGVYbI%9IoUSBqg$JZjG<NHMSjZ%8ik{01DE08X>?>=WZ8"
        "R`sDj9F=UbRlOqhZn@Z&alP7gv-0Gj<y76gQ3U$XLXRF{v8nh`i3ob1!bKMHqV=k%xaJlFYz=xNsKO*s%&RJD%=%zb*u-a8z`*wg"
        "z3d7@uYxRzIlW>kL;giH{z1Gpu=dsOhmcJApQfGads-7l#C<KsqN>Ul%DUS{c`ES5hf0n)R9tS!x--`kk8j}ctD(qj6zHLT``o3`"
        "rDNO;Y$`6$z-l^IDFsKjlTL6(e7Zim|Avof9JYoZ)2|$4^=#LA_Cxi-vvreEda|#-`d?5>0|XQR000O8001EXe)w6WHV6O!zZ?Jn"
        "Bme*ab8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~E9VP|e{b98BAb1ras?O986+cprsXD0svr4Jz!iFVvhr&a4_I%(SJr89kAWDWz7kcFBA"
        "SOB!;xc=|m1;D2$S?=Rj&LkGe-NimX?1DiM{KB?oE=6%i6r(v)B<J)>NNsqgzaTWHm0>xllr%Dv#f%7c14MGY<{762Suv4qN~*4j"
        "&ZJ^G4uatMv**uvSxIF;Cf+=Mma{d;U=>v?g+KBSni{IFb(mB5%gl@zdCdg>z|@SqoRN}l(@Y8j%BA*=zWC(q^G~D2^JfIEs$La5"
        "n}JAvLk%MrWSPw1pL@HOir^xI3*zN3gX+#~qzDo(!AhaZXv^C@1H*tEpWZ)oPf=n;&1OcdQo^8U4F<A2#FdtTe;y<;H>}iQ6!jLu"
        "Yva6oLDs>gI5{~X-|~Vj$UTaExD>&V+N-g|w+S%mx(o{@!YXpQ;QChmh`b{B!ikTDQY=LbK4c;fJxOaq)FWQkMFFb>LHxVqu!@yD"
        "G@px{Z87{TFBmS*L|rlk-uE7GZ*-81E!VPE8OCVjsiot9(14LalEjcoGG}2>oAvy&U}VdRffgvmiCZp6!>$deo38E*Lm@dkAVje>"
        "Q?N`=&Xg3=2?x&|C))CZ*d}r#<9wK$6(d%@wgFf=5eqNMF|-;-3Hj)vGhpOnYaHd8k>6-hvv*2K6$U?n!+-p`34pGsASKr&HQ6Q}"
        ">(7NYs?H3T0;~vk#e+1XI8;BOBF9)VYKYk|!nQOsV1aPVIn$ZqD*%lZE9C7ANKLH-7gxl>42SAMKgK&rkJqc#^G;5MGJ|AD80{&o"
        "0PE@}_D9XM0qOgM!E>m=o#j2=ke-)!;xyqGj8Y3;O%8jNUNu5zfTVe<D~6!iAlfggW>K^&@v8~=8#&-9;m>4Q6%0_)*Dl*;@I%Aj"
        "3P6;b*6>xGUoo?r8EYb>>6{WnqVOtwk`An_%$-+YXMuvPO$s^0Z!Ajit4;3~owQ4iIEp2+u(guJ8cZ<-7fgrO3{q3%-o2yxN7)VU"
        "beWOG(b7)HJMY37LFzdz_?3Xf2Zy%B^aa6<4r)f<LPBs2g@*74Z)kzlAR70_L*a<IIR_mPt$<2Vam7MN47(4Y52Be<9c_ws_8T9$"
        "zHhN^TajOqR?Y+`UK!By+-<K&x17Hw^HxC8CHFM@bhUIrJXkf&(od?J=KAvD(E@6&eYRWsTRdphnI>k1p2FkK)P8z@OM!Zq_<)R~"
        "_`%zQ!kLJ*OOzvm-5CDLiCnfJ|3A<N13f~nr$_oY0UnRXm2EM)Ct%MSD`@ktCxz|_;=jMY;q}#2=m1_mKHsI^7rQ;CoRFW5tcdSU"
        "1cRAF!{#PykyB`3V5)I5%<pws5oxEHk{g_IEab<N&S`526hA*ZJ3C$_o+I3^xo`g69I{y4+iv&;{{JF*fP3Gnyb!cx@MP4SxQQpG"
        "74Ak8k-DGOC&0E2^-g&nEPA@k_Rzat5l91eLyM}Y^{JcJg2Nmz)!lH7?G><|pPhd)Kl^lk{_$QeC5v{Iy+pWkrX}PXQZ@}HtYrJp"
        "1AM(+7LD%i_A&T01mc~<KFpC;62fwa`uR~g{f?C-O<gaQ7HrEid8KHz;aQ6LmEOW6C2w_l^LnrT&2*hMV>igRU@*b(Fnlzg0NsH6"
        "-{;NW4OQEC0bl$3XfxetF+L91J#xsD)Ea@ypE@WAz`uXwAa8A>2C{qTV3)?WR^G;t5n*5-tYP?XRtsk2{vmQN{e94Hk$c;dm1U<F"
        "9Bx1)%$Ew<d()qgY}Yq|*^fi@<m53zy4@E4Eo$`^R(J=21`kxVv9P*^=?}105)H-*Kw|}IK`}1|6Tko;p;AL@R>=&-vzs7Uy4-?1"
        "OnO|0@NgohpOVi#45TIfAjMo~9MkeN%l75+6bfe0;4_t0T)<3_Vk#<`F|FYS!#^qJF<2bdnZ*-7-r=HC3_~mo-ZsOa+Z(sW&f<9F"
        "U|5M!qH6<@z_+@tV7jq~e{9XSF-+FB+UyjM1ehmW)Hb9ua?4?3tBpSyz;PdGJny7x9qR4}+X<irk!W*`U@I)hc-ZLR2A-!~x=k*A"
        "EXMYLr=Na@aP^<&-;-lJ08y--D}~(-cmU6y9yRDO6ctjHFxq|qe_J5Icr>u+k;x9Q{#?K#qCaL+EXHZu9y*08c>n%=_-5gnpg*g-"
        "Kdr1cm-^+U2;-M;qDui3fPfCEfca-H4-d&k2VZ_|`#46EtMK*EBm_!#q={~fmDkY1H9kSS?yKIP{IPRwJYdhZ)85H=i1?||S1jkY"
        "xOC$Y55M<nIS6{<_6U1$jJyvKp64N@z3un7yAb<Ljpq!w?Q}ZWa|PW(>Ep}xbnO<OFyIL)^tf9^*cR-;Zu){e>I28Ix`UTf<dDtd"
        "ZE@P1)&&}<t+)tgiAIxycKs@@g}fEhR*~<&=?sH0qAuJAgHGV6)<<<dHfU6e2xS`2!gndrLTkvF#h$jNwww;J4Y}NWIILnzwj2vc"
        "B64xDLkm<L#w1Zsr_o1}F1GG@GAPHMvv7k~JnJ{k&NADjIrDnUba35%a7t}aPM~%`VVEmXB)sejtWRO(+xPfk1+43PwU?zn<OALw"
        "X^Njejut{rd)nA!#b5S3fczlEz=V=(4O?j1OLD)C9rKrM<~*XTe`cF&`>X%5;4t^nMtJapB+UndJ7MT{uHk(FBeTKn9{4x1$Y*5O"
        "?zN4>DKy%esZ-dhY#-62kd;sH15(%Lx!CKN{0gSm6@V|Jr2|>zzHi}Hq#;6c`_LX?gCg4Ej5h%??=MhG0|XQR000O8001EXHbN|u"
        "G!*~<(?b9NA^-pYb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~KNY;0d}Z*OcaaCyZ%YmeNxk>3sYKj@STqTclTH4mdd>>6X7v5#!*-Q<8q"
        "Ltsl(w>)Y|D@x;;oy>nvRgsh^i9H@~aJU9+OMDcI#d@=3*LB}jN#Z1Nz*E6PCr(9KR6!YMsgq|}GIaJl4UQ*P+&aHitYB#wr`%y_"
        "=oCE9L|kUY!$@RR5kPAd#^u;`-Pf;Pzlu+JR+P@+_c*U#K8Y-?Urww%)-NS1qB!C8dsQS!ydM|*w~CAM^{c4JPEL>|2`AiyqsG0>"
        "s<h;V8ju4>n5H{_rWI7*u{s>$^boOt`^TytbByrK53|o{lCiKJzK^3UV99}>PHY^ql8u9^U_=)+(|i5Of#1i!{rchUjsM~1-N&J9"
        "d7Fj&Z+E;%SpGf@`MqkU`P|Q0T=3ABBonHS-y=w$C@x!&8h#ibVE3w};4Ji6QDmQe_@ifvJc&y`$kG%j18@r-WJRcZ%j}k?eqLk<"
        "4%!B1cb7#`0w?Q>sMH%2CEEiM5m1d6c@d}O+1U**=*fhw+@wU8_{EPy4jfcIAf3Ov`RVQd{P^vM@BMf0f4KSRY@A1lwR@~iEai7B"
        "iQ;?rT2q?3_VFPA`cCR0ISX#xp)ul$EoT?&d33R0m%)C%*zZ@Dv)TS)5v~~9&(=#G1<~T-vNb7xD34h>DS_OhDgH)TL=UrZ&I(|{"
        "QV$=ff8BtkF;F}%I~b-_UL=#ZEPY?zV6VB%BA8x8i_85ix>{VWqQ!i+i~`QCmf>=}UIrIeVKfaFYu1{C6#gpA3NC&_q};z(MFI4l"
        "e7N~I`nx*>oTvP9HQmpGt7$lcm;FV!*w3RyaK+e~2h*!y9!!@J!?C!m;N=*F)<xJ)=8^sS;Ro}?@Nj=)u&8@1%UoPfCWp8L!5RnI"
        "X+p=8RAM4W0gomeKu^@aVxpyQGM~<uqv?7yU#PL8riBn0AAWc111K^fkQorZxpcukkLB`};04G=2}B@F*opm~r6Uo<pw;*!2yp6T"
        "IStmA;e3A)usOTrj9;z7tLcmdms7rGtL2nkF4<}kPUs|B%lgFrpuNWTgxC`@_Ul(6kDQ2?!O^ePr;j2iydulWq4Uj9ist4WO9UTW"
        "D+z}_ped>?8aHT~$EVvcE<8AK*oxTvQdE2hS|b*vpWV`{8jLDX;K4j8J(3m9Eq@p~9ugF#@Gcgh#fGF|Qg-44Q2I_s7RSKZC*f(t"
        "YQS3nwbLjzefaAr(oU~bV1PisS>s{pB9&1-<b0#K(Y20}TiVon>yHUoa72AtY>3DcV3x=ROP&h--RT`PuY(S0!C19vh{AD(gg`un"
        "0n8c~q`p0O0s=Q1Y)7+QGdhl(6b9m?F`aZ97-7p%*)q_&8#(SXaJ?_82=t~$4f5IftK*`!C}R+mK^sQf(vIP^;AIi>JMO{ruvuep"
        "W^Gjd30PHHaBzlUax%s8w92YTWp5)a`zlUCLs()ooCFaphbXJ?0c~6qA4qltSQ>UfTUs7BKV@knL9h!nudpI8i`U`mf%C6UGai&e"
        "8OV)&wB*3~uT2fhpn1V!!JU7wq~bSVhzrmC7M!OjzT<|Qgk%OUqZIJO!OKa3b<r^HIWP=3N#Q9L$faYq%SY1OBP%@`>c3H~!E`kR"
        "4F$T1U4+r)Rd^K!vuGA9_MpV4>&pcX!}*l0*7GZ9T15e4e7OjDFq^Hy_2sN(Rg{r%my3OH6;0R6C0|Eux>#PYV09S*aK2we7yFC("
        "Vl`i_0(P<IOMrh7tmols#e;pcSkA+_2Kr=Sc%Z4_$9N4#5P@ZDho$=@ZkR$-);}_F^n)V<$0WRVfx7{wK1O65IR35^=Cydnd{Glw"
        "5~D3b&N0|+OXT-er>n$spJc&p&1LQJs?R5v$lWRE0s5&Hy^M$Lwf&u;=~|f7L^lc`B%lhsu6t$D){sH)E6FW{Y2f(adElCxO~=J@"
        "I9MTPHeI!j4%D)Tu<5Wgq%R&<mFl5}IMen!C)JQeD~Wo+W&n~Y9^%%KQar6r9KC2ctb+^MWF(Nq^lV}L)&)QKl$Y)f(4hA{gai*j"
        "#iT;N-rWry>o7QtERDh8HHPl`Yn&r)e<V>`q9m0<y)=s;nCU_9XkMhrry*^(O%x(PmnwuBO2C+fp%=N2faKHlV+?;jJ-HH$S^%vF"
        "1#IIihog`zrS%V>=oVmqVLg}vF~P>I%4l@y4jd*Nki{_J#?ol7;Vj5!BY!}S&@K%M2Xy18t#ZD#a1l>H-?R~wSVj*W7C;#0Zk6QG"
        "K3l;Cy~&N>Hk2?AI&4Nhh@-qkTuRPT4erjO-@3Gwq>{u%aFV_>zq=UcK#gV<`40M`$I}v>O&6Q3oB_mxQv!||)3I<*AIdo-E&i8@"
        "wuE~9NQdxb^N<t<_4*JcLgc!CD1nQeYezTpG*pedF<2lIweyZubNqh=g;m>S*#t^xHl03eji1pyii=z7xZ!P}FEy!{I;jpqDR)ye"
        "RFa}49qk}wN5a60;0Qi}f$E8$_PhX)s%w1UrEK$b=t<Ex#3*^fQN{ShVdn2o^A&|XEw4?Kp#4t<Fn(VaU=|TKPeL_DGXXJBsVBvP"
        "oJ#bWI$F4qIL)Jo5pt-U+Ex6Zkqz+{9**uDCDM`nUC?YNUNV$+;9R%K7PuWv_b2DqTw$u!45AD%1usTs!s(b5B@pWrmt*Hg27#6n"
        "C9DL-<Y5;gsP8a_rMc1}N+}HKOtT6_{SKR8DV93yw2u!}Rtdl!yo_;wM}p+Y_hq8P_dVE;I{;66?-lN7I~`rIQM7x!7%rzz|Bcoh"
        "{@^^I(`PvE!QiW8v&COpF0b*Zt)Y<3ejN&!6qZGWib6Ib{{i0d;Yhio5;%x3{eo<2%X6&K+nzmEvOo@>hPjgqq00jBx8R$6!8+Nu"
        "OABgyh0n-CGDr!1YN#D8{d-Q0%-8FcgA5L0Oeabn&hldRQsvT40YG4SEBvhR8wQn1r>ku`x}vRg3f2)Qim_`O49@y)FebMTDeLvD"
        "a1+Kr44i^iP9Y=%pcqyJWy_DGl_*JqhePNCVe1AGQ+hXmqSojF#$z35LgMiMSVzMyji5+*eE|7oRvvkfnjmm}#?XI~xsM(L%iW)V"
        "tWgZuQNG5`wQV9(ZdFp5662jN3052K0Y<OIm&5=mas7xJ`b!-CWv6{L1ZVgFJ!6JUqWx$%j>gw2$|hHbKRR?tbpb)3d@!TKHisLG"
        "P4WU$1i<8>5+8GBEB?VbF*nAm-qwsdtP@8zU#(&A<Lz}TaJ(DHcxmX|;i-{OXm{+%1eJ1D_$Ranet86X9J3?@c#^bri#Y=}nRAST"
        "`8}X8q0G++!8n<W9-Rz*G)h#4MwE}VQC?}pPI2;JT#vi~X>oGYP%KW-t^n?WpRh9^{|P3f@-89ZxOoUJD6G(-bSqOr$PYE&OIxDS"
        "=|$OOQ0VfdbyD(%fq1Z&MCn2Un{fc@h*mbmRkuTQ56~no91#?pr$CtG;f*6wmIGBW-2uFxlPy*TC(Z?&TLN}ZUJA)gi7GiC35(Bo"
        ";cyQ@5SSx;DLB#GzniyW;X0mH7eJmN<LWZ&Os@_-EL{S3)pGkZepwVjt1S-g#^FS53TU^}zT&7wm3>B2Tw5poD~ku4W{6N{G%6}#"
        ">7il=jx<uS(_o<83#qA7=TUdjAos}vm&nmm&=BnU8mc`dtD+H$=0&ahT>8WU5H_b{JDpZR)elW|Vv*9L?#QJ~H8JdqH`ACHLPW{5"
        "ZDY#Du_UqhVKl(7e(EG{l{T06p+JP&-B=*sdzJ<8N4zdMeGe4zr3*_(gn~m4MF!G#@?j}ZaJ=y%%%9DE*a<)~n;t-@coGU5ED;|-"
        "8C3<R)K~Y)&(rq?l&vsuuCpbC_^#{KDv~z9@#sLlUW|BH<q7IV?qg(u+-}1iAd<<f*Rc%J`e54@BH4;02mdjJ5CeqBCX<97Dvd4v"
        "eQt!7Y)L?##D^FkWPZojO43)W`Ughwavv7I4|rZWUPFqZ^F4o{Ckie(K3S;Pcmbbeq7JidxMP}N-5H#Z*89Z4o>X%%Z#HoL?9B0&"
        "5%XD7%W`_6_vrR)4wzHQXS=n7fzFX9-v&=DWlYvUCjcMY>)7XaLKIhNUo^;?y!90Q-ka3%PH3=9D0N4YA5wcJrdQHd^*yveek+6x"
        "3bpjLE8wHSc<^B(ank1sk|iIGLI+R~-0%Rs^9Fd^IZy=bUZg}|wyByx=`uwl=?bVEcKc#d7a~v=sZeh+aiiZ_9)xUy@Hww^fx<U!"
        "7_ckQ6Px*TdD#Zn??Amm`48E%Tu*gR%P&!FAn!4{OW>HQrF5@Wi%1Bv#+Yq3m>1PAUBaO<GDt#)*kSSqRG1FcJ6P{BTg4di?h6H~"
        "j@kzL?hO;dV;ZXR?AF|UM{f=AJn`X-_t&;-<lft_(2T&zdf;acRyG?qCbTbk+=H>ahQ?b|T@)i%16jhxdL=Oyza`+O*A)tSSGr2Y"
        "a4I+U0vTfiTaUC#eXs-5bu`LwcE%oJqX379U`8h!O;vlpjCD4W;>{3>F$nJ0QwHyJx<nzCmJZV9Ip4_`UQ303Xy{7<sH_jU^%UxE"
        "+Lb^CQQP%GG`pn&q&Rh3+9KY1>(5i8T5l%F+rr;|eZi)QetnhDZ)y`s#)LtV39h|0QqmB~^Shy}t3r>+USH*D8b23gD;5SD&@U1*"
        "2Ka3ggTe2%b!{pcJ@0F+-LCN{jU`$kaU=FkfC{U06-X-}*Fp*7>SuERF%#~qsVr;ugapHYAOCGl(k2j`dW76H_JBrkM4+>-DP=)W"
        "ovK8}$k)!Y+rKBXDdOL`ZFZGo@>fWkNQ=NIEIu3uJWf1JmJn(K5YT1cj8{1??!Y+>r49G%y^HGP6<bLQAD|pcz<!4wU;z%ME03nY"
        "qmkgBfeak?N8rLV6Kd<&Pr&gB@a<976@rKHfWC-eC$43l071M1PWt^DXP<#xMB&rIE%fPzZ=CnPNCg5UE`W88>@LQ+0=z}xKukp$"
        "GnHBoSmH3sA2~Pk?<P0P?^<;Sth$durGc|pWC>+nega1auj>m;^^1qJJaSBwn5C$wH-UAA6rvf$hT4O|HZU8lg`z4AlmY<G2WEWm"
        "HKwpC6<)RTEhm=75f)%<%z8}89N-g^T2Ky;lJ_P(s&&$JUte{+hYs6I)~Drw5pY3d0dU)uMX@CLr6j?yhIMVD0ADqx!za&6hTPe<"
        "Bd)-ImeIQo)TidP){Eq(YSK1->ItCNcd4po6c`-rr{LO4pci>-COgL87KaLB;-9GpHYNgyw(*oVz8_|R?+?tW7#f0?Cg*tWXhdaT"
        "5b{w?85(ZX_7MXYTAFcJQe8}FlOQE|C45ZQ44#K8r@a_(Lt8j+|8Djr@U`sOHlXX2+Esv7IOJZ8q<irkvT}s1*VK_6PF1y_&n#ld"
        "-)KY?kgyITIo!DDP;%IOg*!O%B;U9<_Za1arOm57JUFV%&(URsf^w;(%$(00@2kYvZQ1cNTcvcUab;*nL)ZQkbl<B$(LV@Gc#VKl"
        "ZU~yjw5exrIDUn9wRzB*4y#H~!f<*{9)lMVp51C)5C+T9Qj%wE#eZ_qcc7eas8g-{fkzc)dvIAyb$u?t&vcmbtV62kA2+vBc*yNL"
        "CtB8YLSJy>M01EQUsR;pL};~w1dqN|CP{fG!YZszx$GZ0QBsMc$>jL+n$^+2)*uz5j)?#g{WiQI`F|-yp#w`rORa#=tN^bV+Q8uO"
        "X6a$4u@x5At+f>U-ATRBtDDT<5P)v$%OV1;anEnYM5l$i9pe_Z-<U%;rd93jf%xH%rU<%;0OOA0hj>pluyCC4cb)<FkZS{eghT$0"
        "s;b~l{RsyX)cf{__sz{j|I-E~A<8=4{O`{<AKw3X^V2W>$DeQBemplzR)kB+wLTa@5vyGgzoxBbHp#@QwwhiB=8@<g<ri2FNRR0>"
        "H)`dpHOgD1ITQ|Z;v44<{3-MXz(Q0o4fHqAJ2++#Oo6&d`DX_d0R&8N?2W#8v5HX0*11V=3-Q#iHKo3!Kwhduz80@e;tWlOy~w}T"
        "N6lDwXx5ew+Dz>|V!H4^-f+L%Rmf8eU7a~KgFsuUe*v^<M}(HzwR(0p3<IgfZ)ze8;<oDJ6ZBWkeSOk(0$q7QAYtg6q#Bf}b|N8>"
        "l})|*DX44G0yrz8X`{#Zdb$dq^y+J!P^0QVStO7ixVIjd(4Hu-XC$<`DL>f$*62rI`fCJ)Ndr54)q^o+dCpT?@y?gOh#<k^(bkQ_"
        "9vRGYI`esioL+zY5MjnhzMX~kvM(y|*L*QibF=-E3H+V^YyyA%+@|?z69S#nmwZ3e4fVwW)jwU=w9$g00@6k#bllMu6OnTaFnc<^"
        "lPgw&<l*qK&LYudRnyoj<e=Nk$Sku{-%3Vf8Ii}KzMz%JTCP@P$JU&)8#Y^McVM}P-L9PHw_(?AoISEXIX4hofVRiarEpht9|872"
        "x|x_fIGY!c!8$yHPp%T_(vQbP&A2%WRX#zNqIA#ODSJ<NgvPJgzkA5CL&7IyZb!?}&HOvO>ymIFP?d!{hFnvf+w&lnUjTtzMoi?^"
        "`;|WClk>0c{eUq>d(ku$FT{?l^zqA?1dQkSQ=EQc00BRfkVK&SflmE_@rZ#z`rsi&wzoU_NCmg1?xLGqAYt2O&0e5l$q4*XA0{`>"
        "9WN^ZLzxQFDP=r+xjW@l2|yfd0*p-_#;#T5jSP$Zo(g}KyDI!OrquM4rKs3AV)vq9qquFrdY*XCmH$6XeYb#7#;N_jn$7=%eE1m}"
        "EE`f8;b+Hp@90J4W!S$bBuQ=`zm3(5N*_yajI}vwa+-Yn=gsIH4jcVKPRMLF{r*f@I^CJ*oI42d^Dt}{5|J7aQlnCjyj5;Ww=#D4"
        "vMn0*OST15zK<(G7^;z@e^^Ia_Gh1K-Pf1BxM#N4RU}q7>W{lW){jE*_j1qY>}Q{Ph5v7t$FL;E|Dw=XBzf0Y;CCJN3mem#>8qx*"
        "6@TZCtAlAz+tGM1Ce6xQdqHQf-l03|3k9UT`Oj%HGv8C6i~&;C+{piZqM!I_uVY4p^5W~cOi&L8o3pCZ3*@L;GWh7O*7a0$_oNTq"
        "Y~MW@x8s(iIcR0yM{)3dDk<=NOriU}d#!CNCe&ZQ`ZrKZ0|XQR000O8001EXzStR2e+>Wt4=w-zBme*ab8=%ZVrgb?V_|G(Wo>O?"
        "FJxhKVJ~lHW^8G0WnXh<bS`jty&7F}+qm&PGx-lFA39_r(RO+tt?qOW&H3gs(_S-onZ76*4n#s0&lJf3q-EFR|K8mNKmrsgE4fbn"
        "U`gD?Vt29MU=RfFTPax0b1Aa<HZKGdS>EL34XdT7x#Ue=l}zn8z}e=3ZK`I+RKv@R7gZ_PZ$EutWz~pHRo$vM2!fX{UcSipbtM~?"
        "<=btMZ@S+*u6FM4-&Iw1KO~T>_We)s^2JtGdzMy3AyVW*F#~&BwPhouL8y6yR4owYd0(mg{yjWLW?sI?#MVlcNRd`D3nf1+fWNev"
        "u~##us#d0A0dmQR&3_=k;^hklfB6<TJH$7l2?M=0X&!13L~OZa@T7*Vno2$-z(%@D?ydr}5dWlTT3Ip&>zIv5qi@8vl48dA79?ZK"
        "e$%SRlC%PO%SHv!tOu0h-~bG_C5X_Hwp<#{V_+Oa(gKqR$WhJH+ieptQ7F-`Qtjq<(0oUGiwDheh17VNiqIiD)4Hpq#_0hue9P-F"
        "a<#&9*yi87XvMXZl?;P@t`us;<8x|Pt0`~89!v`Kp?U@WGT%UMLOaQn*w@WNQpuz((;Y8w1fD}CEMsVP!*fFanPwbU^o)7us28$7"
        "Es5+&w%Ss)HCj6GBSp#gBH>w<WTFsFw|UCTs?1Yf0EeDK5EDIzVZT$iB+gaBH>xVyMubt106YhA9Khi^xDX*4<>P$(NH{gQp9@%w"
        "Kktcgm7t^Chn*%oL|#6GxF47l?s~{UoFE3Z!I?4yjIb6RQGm0QAff`B2k~evKC-Nt1*h%RJp){1)n4SKP@wW6SB;}u69?nA%(po#"
        "!QT$JcmG(`McxpV@K%Ugl$lz7kgbSjJ~~<N+1MB+(ZZJ7fL}Z!!%x8grKt?UW&#1Xfj#%j;A82+tiC#@NYN{{0m{hc^P9Y3c28z$"
        "0|kwa%16See3HY_x+e8{X%M#b8KaE);6Gw0qku&Sufg!jk004D6$6K&N4#Nx&Z;d#*}6?J*Cm*WfKQW=V#sfAzWydOv4x0T2Jk_8"
        "j$l{xd^q=JJO67CMe$DDXZekQQ??FQ?e`$+65TuOTw6a4$ltqdk3;nYHzQH>3{LV45(3(pJf_%dH#8aGwdMS^4rs3(^}Igf@o0Z@"
        "_{UtB%=g&u7}z+N<XNzA*mR63xRF&`15`8RLOQkp+I^o5$h*Aa9)m|dy+)I2yvN!A0AR9=10q`hh)yu$JNuj3Q*)6abb)E;YC-Rx"
        "dJO1u0f_$Mf{ZndX@@F|VBuSzen`d_1RfJbo{Cb5;R5?uFKNr$vO1LA%BQ}V(xssoR5J=G+3!wWMKw!O<iPXt>Jt7UueQ<<_C0%Z"
        "GUomwpa2DzE(BK%YYr8zWoEb7nj%kAQA1dUy@i0P1Uw&33uE$XjnBY9XSRN{wu(_GFEepZjAASjWxE$p`U+PfHzfe)w1WMx>_u4k"
        "5pyF2zwIA2@?n6GH)0Qr9hZbwaZCN-<S=oM;yHR@Trq)P2DFXbD?DF<M{A6wS3VY&k9pY&Uoc>6<@nngHG<ll*doYU$U`Y;E74Aq"
        "XiqtBLnItS0TUi=>xgYnHRd&DIl}b=O@HoKyN8Cis5e>4mB$9_8Y>vs^6FclAnuZ_<mAH3%lK*rXUFr>SnwM#V2#uLcI+8TzDuxW"
        "mj?X2LXX<#W#~|Hh(w5T()!<(cf80mEyji7`?|oy2WxxVr{IaEuBb8zAGLHV6IIt_==vERsgeDuuLcWIhMGw91@4$Z(UX^HV>60j"
        "2RG$xCyH!dwT%{_V_7$S@vrm|;t52Mt6C_RgyqEMS8!Pn4Rz?UIcIP@dww6n2Rs7XaG|ffbBH2C3?KH&f|#IxtQNG3H9R^h3>;D`"
        "&e0h~u3oRY<#EWaR}1H$*WIpXQ5#a~tYqz3nP$&)IDm8Vw63w2(Mv9n5C0UnQUa=HBm_n?uovWq7Cx~*vr#AzeDb}K@X5Z98}L1c"
        "Di|w;L0C0QSG^dIe(ZSlYVZL6ZpFg_4*sx$FY9A4@(2**GCYRjpV!pSCUA5dhNy7|?Oat(<CVNtyQ)cay)}tPZTmf-2{YOl!;Fp}"
        "!O<IOKx7~MH2c43%Pr!jq|*j61??mJ6vj~{H%+RU2AV=@`Ut3)ngZ+EX$oo!rRPu0J(j2bEBQ`!aDsq_t(}C`@J%6-E%=O(wam*V"
        "yhD#3_{(I4kHfV%YxlL%>t>V|5;$MoBx1s&`CcW}G7M(H3_EnFjzF>&lekLreEAcH4(q(K-!guB+GV&8g5Yfh8|?+F$^t?YmBsU|"
        "6k=|?Lq`#uM2dUtwH5WTtl0_L!WN-Y4;_)HH><1cL?s|P!Rc8Ks)$CNhSwzJCB(x-V4;{;PMq0JKP5AEhBbCvR0pgp@m$^3=Tj>s"
        "P`Z#p%|E$T6O?T&iwoEcN}9~K$6%A+pw=&TEyPUZ5XW>L20biVb%6@`#4YUUq^uh_%-x>L+r)&m)1taFK65g&uFntMjmCzJvk?SC"
        "WgO`BM;5~$sqvx+ow3@znW;ey!R~F#MILrfk?e#7(RpJ@$D*xzz%QYqQDX)%65QG2Q>1}6I3{eB$Z!s;NE~vplYBV)TjF&EG3X(|"
        "b2A!4*Rkx_Jmk(Nk8%?ltfyBg;%!?Ld(=!QgVp8yZ+yO8KYsJo6JXo*pRKZPV~XWY&t+4MpH#a1(Z}d695ct;hB<lAo(*QNRt>Oc"
        "<PV1dLg{eO-JB&_Evq|G;syg%l684=N@pGxoIhN2Ezfg|=SNwc4RpgSZSppjkIs(%$V_UC;PDvf)hldpj|VX$Mjng~W~>K~OlE?>"
        "C;Y#i82!<qN!sE>Ap_T~io~diCeFqh^rK^y8&#qi(YPwEpa+Jauz@Y1``<y_E8O^@C)&UWI#-T!j)Lzu8DALGSPGtnSC?<Tg8m3U"
        "z#%q)XOulVaa(6N2O)y~p&EqI8lb2+q)<oTFv}sa;^{gSz|U1l%^~FV|A>-d@R=jBZcn~rcS7dd2PWY2fo8ZA<4p%^MzuB|_f^2?"
        "rJ%VBB=#!`jC32)bERRnIRQA;Pbh~9Lt1jd0t=222<aX*nV>a><O9IQvYX6y@t|8o<^oJZfHjrLCf`;mFK)zs&pmYw?-5T132&jL"
        "Z*<XDkJvo>ob)g!dX1EwkGBhUrz7?ZzJOJAgm4g}Lb_7CC5c?3Pb8%C<fKHN6n29IR*4&#HxJ#!z=^(SDody9(n&G!IP4~&ve9|C"
        "lhmzUe4@_a3GB_1T3IsdL7vx%$ixB5gwqG0UP=L_D#&B8K%0r+cwr1lN-G@x!Ks^q=iGXLv}HN5#CZ#)B;<}yE#B&PnyU^t?|3wA"
        "2vkv!Z|wxcNz3GXIx*|Y8Imn@@?L<zYe!s3sE}aDA&S7uz+>^wI1(-=<LGsZ@++fS0zaFNeX~dPb;h#5bVpQrA{0M826P}j^4!s&"
        "N;09b;|RwI4S3NMdG}a)X?I~|lH}S`4HYLu4HmBC%Bz5!$qM1DeMOo61dc9Pujl}tFI_xU=}D-*e`!2Jb-Gg8c_{xDq1yO+N;;Hj"
        "OVI0|Z@|vv6iMI(yxC`ro1yA}bz|6{zg<7a9#f9vGc>P0-h7n7FMM;PV|lunt`6f9qCs>RA&P%APDWXW!DEuVD4hOlWM`+?He%mK"
        "PHRSEr>UXSb0a6)PfiVFOm#Q~)&=4e)T^w&aGZ3;gm9qFVSvNG=dz;dK>Xg57D8+{!2kjjIj>i%A$-;ak-kprt-#$!-z$U`oo+{T"
        "kuUqJ3+Ja(()}v%l_4t9!Fw;ddMblTt{K*B7%wo4;@3TpE7(EgH@il~wpRN)hc234s@S&*Um&{^SikMbYH$*yFcH~-5t^BPlg-SN"
        "*rv_kBi6*;;#)kp7c5l5@~D_elnN?Qe0AW#PL{G6UGm|L6NCeJj#&eC02RfZz+E|k@0Rojf}U>Xb+*sP8qD;ZSM2*GBPf2k`pebj"
        "w{QF~tbyniCl!BAwP3G~P)s-qBLQowTTu?Lki>lr#|J1&Z+>NXfY!(7X4q{R<^a!Q)9d5Ri8{C*O-S6{TKN6-)tkPX!VjJ{Exp<_"
        "UFDf!63VFDCt6;o-Ziu!O<CDbDbsh^;QdgM^s0}L3x>rWnmRr+AbkC-&5KN%nQKfe@ab;BjqF)HTDG3PvwhN|=J_dhmJM68v<Wp<"
        "H0JEnA>SF#?nHWvcX>6wSZ2B|U?{Di9pg9RJH};`Z($c|SiyZZ;4R==VD9}~WYCtm4NnZb@R$M*UXt2Q?fTb$zWeau`p4w$zka+<"
        "{_*bZ^)J6&|7HUM_H|Un9$+)fz%)M4C6j(+Zb#0T#c6yrkJUOdrUmgEoE=|{93f)HrC^+$>MXt{Vh_Qn3Q9$LtKBh4m;U&Qfm!HZ"
        ">S^(?2p9H(Ok~~La0J2oO%F<7eA+$v++m^>#vZLZq#N=(GGzk{Zxo`AW(l9gg~@Wm!qk9mak}bv8`i&mZ)g7u`fzAt?rtZ|Zb;_%"
        "g;S5P8LpbbXL9-JQR^5*8rObN32{QzI6meXM5ji#iT?#qO9KQH000080000X0KrM{K}!b!0QM6A03ZMW0CRF<FJftCZ)0I>XJu_|"
        "VJ~E1bYU-WVRU0?WpgfYdCgc`Z`(K)eh-lUz~SzL)Yx*`!3+kT(YAAGivn#EG}#3faaN@zD&|BY1(M1Illkv+&Y@(>vXge-#w}uz"
        "JUVxN{AkYknYSwU?BC~ara!Q^%CkoJe8oi5R4Nx<8O`9oa;6eq7R<`VIOUDKXHJX8t&DdG=lt;C@Icj#v7Q;XezbD^=-OptO)lML"
        ")yujmRRuDoH8mPuRcgt))l(Qw*lBC!DOkM2AsQjkr7BG>sw-L7BFT;RR^;CGZ+=?)_y)932II5NWmI%{P{@)wQOXR1%FujtBdS);"
        "ne#Sg(|_Uj{P2Lm2Ul6T?M{%6gMcbo6rS*-jc2sFnKG7?B+6xC3F&OTjN;v8;WG60E{fSV=^h<CNjOtF2sgrdH(I_@{z-wwf|mDa"
        "o<Fi6^nH}Cgay}KBXV#XXN@grHfBGuZ(pzyrf|4c(Zv?eH@d7)PO|^t>T7F_jd-hF+hEaTk!@OwYQZ}NR{GYi-RkQ`T}pB;qUTM~"
        "eA`s=0!9;-B*|rGAA0mGn1a5$^Q<<7D!UGMGf@;OJQcE%^z<WG{$1wS#9GwS&6Y|Fdq1m`zV2GSt2-83Ii;N!8nQ#~OoBU*6kMgL"
        "1ZGqdJ)IEP5}xAE*9ia%Rwhmgo392fX8Hf3x9`qba4VR2_$3T*S2^|&JZBHNY3|vBQ?;suRsJ4oRLHxB<QXJEvgrAXWB5Eg{~o{M"
        "m`^rFLM#KLGWjt1e+wIGWg2NI+kIRsDGF&F>?+V&f=i;RB8Up=QnKk3`k0AXv^YD8*)n|plTSL2$dP!f0-d@iy{e&umA=YAcZO!N"
        "@KWy)wG;SraT&(g)21pWOv_vB>s=FI8w7Btg2-V0Yq}k4!{e=7uUrDaD|Hc#a0!gW@y4mf%2M5dUszsnY&(I`bUJ-l9O1v1Magd7"
        "wgVs~Y05bC{>@CcRmDe<Ai^p=L<M=*z}A5C0A3R_j%tEWqzLSDCM$@ZFK2w)7*q-k!e&5dx>XQduZ*_KC~%a$crm|>!!8T3=DwKe"
        "&~l+(q$q>Z*P~rkNgX-qBT^X?-r`gkqAxi5qe*h%WC;}j+@s{L-S~6#&Xc`YwBbaRnXwtF%yLszs=$8Spc&t8>dnT7y1shFri<bz"
        "dO4rKSNxJeg@?meuO6@S<9ne?RjaQ@h)X!yMYaK6+&WU<C~I^rwT}pL3!o4Iap6yoxF7}X@yGR?-B8}HC-4BNGcM^ABnl{i0~KAg"
        "*Tb&h0B#wN|6D6e+NOZ?A?Rhlo_u`s?rrw*<m|s^=dVAc{5ga^Sa|#X<d>h{r(g<t8e2N7p1=O|;oZld-)E<vPClHTr~G&nGnm<g"
        "iX5oJ;>??@=Q~mVFyF&0x&mxYqg{~=qu4&d#vnA>31xy`p#@9KWA8>nXn*-PZ0p{`t<J{cuBl&gE|Igrnf05p00NMvC;&;yJCZl0"
        "<V<<Ygd^w}x?`YgQlg546meoDP^+6QW<?1DTe~?M!AKz;z+B+GDT_$?j9v|?-=}=AR|l^n$+Hp|v$~PcI;1KdS=*n^t(0Zb<)Ya-"
        "Bu0KB!Gm3x)YpZwQ5d-N+_pe_<(+aqGuJeHLd7jm0+}uEy>t-;i*S!hrOtsLz_I1a=?^>}sK^ZvZ;j}G!I5uH0P}PFn;QTg(R!2H"
        "s!U4(X8=yqP9mx4?Fy_BjZC$<6;|`h?Na3{nO|eAq7cR#R5DH;RHCi0^qXr%Vp|>Urnq<Yh(#A*p0x8ztXVKb$J7O!FGmo|lwVGE"
        "RVdxGJ`Eghl#@Nu`E8M6z4t5ux}-b!xle^?0F7WrwH_@~9(JX?swGII-yV+^ciS_hD+wTn0enp_?uy{2J6Z!amIC^vm@^L9>ufj="
        "N3I3f9p(ut4?tEy(Rq9^|L*wsa##6q#}a+r@GzR|E=aVCi9*q?O6*l-mJxpj{W;%>yT5R``x<F`^mThMM3`HNH{Sl;@g^K-M|bzC"
        "?tp>h*4Xj2*zkmJfm1&Q59<u}Tynqe?OXCALip+2A9*~us7`tjQFJg8kLth2L;UQS9!&j%?H&%y+-k@CN;!ZxL6<X1Yr%jxTuJ{U"
        "Bpg<}v|3c%Q3{|4VzmLOWQ&Uuz<;y<`1m5j^u2Exs>$H$tFL4+jI^T^y98iOAQC*?C!0fQ8II4oR=^_y{h~s~1O5fU;g20ztKs?z"
        "jeysf2ye9eezkyG>WQxInaFe5z;z1ZxMg&|!kdkjh>l>zOPZX8pMFZfIKW-%Tx%d~rxi4kA6$Cr2A$GZcdO+KyU+vj+AVF^+~?1q"
        "`7o<Q#=E;8@G}|$$ptYWwCK%4AVn`1?%6^|$+MU7LZgxi=_ZDI7H`!vsD#$|2%&wkfJ)y8;3*xyqo9Y?>p~=*$(1#26FuMN<~tz{"
        "F0TAbx8wfK(^G#;^xzswsUa8oT*5}~%wV3wu|)z$3}>F^lwB**)ZxKjP)h>@6aWAK2mk;8Apkp++D~2$008YO001KZ0047xV=rQ9"
        "W^ZF*Y-eR{ZDB8DVRT_Ha%FIDb7^#Ma(OOrdEHuBliRoves`7s1H-&fyAns5(^g7bo7#9>TS@KV@opX_6Ba~*BW5U)20&?KXa0M-"
        "8vqF&b8PKCg;n;5K%>#<<3l$<-}isYih?JeENc;SPsSU*Wzu6s;=Rw5Cks}|O{u&sQ(R;$&ptDim4%m<!Yk7>&kF7dUX|X}<;T$X"
        "{nxKvzsj~%DU`Qia+7E4=KVjiESeX=o1d~?SE7u$l&@c<qTG6MnddylX=KROv1jPM2ML%eg`rBAVsXx-<kC_!O@^dm3MH`JKfx=?"
        "M}6xoEn}A7^6i#|aakzAV%2c^z$=!CclT^t<$Q{-rOaUN!)4ZezcLCM<BH-CYksdR<Cqm?kpW3$RU4t6rk>=u=14T0BQx?*&agVL"
        "RwHuo`jrO{tE&7EQD3CWW|pb$z0Fj_i(MwlV#^ELs;N<1(zHmHa*~Bwp>cwWyZZ5$58u6uKD_(&`>XGN{Nc)5d0#ZWpJw-}7TjNY"
        "f$xfXBoBq!aFxZgc4;$9J#7Zgb_;w1D`bT~nRdBGe8=+N%bIvJoaLTJ(9Z?3)os+CfzCydZ;!72`EGWJ1hbDQ(`>Pr|J|QDlyCV-"
        "x;QyMU-RSmEnBeTlX-G{da^#9&)=M!ot{G1o3pd|qBBUZ&$M)e5q!-Hu)^-RCsD!ISuIH(Lo@sJxWjZYKYeqO@OgSRPx)EG&Q8zX"
        "o*rM!&)CIze7*n~E*31kNM9g{;PMkG`sZ2*aFC-9@2+M)qx#}Ge{(ipFXFfJWC1Vhi{xZ|oSwki**TBrZ{y>5ewwn5svpp6UcdUb"
        "3CsX|`7<w8A4Sc<rKrt`rjJbCEww0FAr&iP9%aeWlS&xBB$ygBfcAEo1i)~5M47vU=pxSRnn?~3w%umRY0adp-9d_pTds68In8+%"
        "^FnqAxm1yiMOJBc-haIxN`S@{{V`E?P@A%F!|zd2!9UfR;7B7MCCXWGTeDj&2K4xP0$=Mq;VG>d^aorrbqE>_Og+GHSU<2%vRGYH"
        "6a076on_+9zQJZ=5U^>DUC1T`*~%TS1cw+3{EbqTTpk?(x@_t-K;HJqpBNfhr$iG%Co}@TtO|VW&&xd*!Nh1j^@<X=g)pu2e2WSX"
        "gn!LupXal;Q8>H#`pA6&FC%#-9_&MY82aEZx@Fn=IC!|^-mlpDPKZ(jeqG4ALbT*bq=hAr^NkQjEwD)Y3%GxIFrLPF0X>o{EAxH>"
        "n}N{_OmCIi3IL00dU3eZH7*|+jQB+HA<nrK%+4ROBnkAw+YW0OGygTc-*nacdsO_a)L8+9SrWC<Te$@X05*o{x~19U6a;*P`&%T+"
        "ec(j*6z17$|0F*g$B5MCEx0gHa({z*#u@I4a$mSxuI6T?N7MpyLj-{|24JRyE1>QERM&d!jRlejp%nc^3%FVtaBa~DV~OoU02pKV"
        "TI4>i$qwc@NbI`~O8DfCLzEOX<w_$%A+1{Hd>a9hDgqPBhXBpxBg?!-CK~F-SW&wfu5r^H@4#szE8$L>anTVt8DZ{Vz2C(HmgfQJ"
        "-Sh`L!l?<Rr#pis6Us$*XnumYsn*g|FLG*xojUSNCQtgs7RG1AZ8UJxjOR(?+L(Ak*5wFiCqfCI1T}~-HcS6HyS({wF+H6>_*TfL"
        "<+5QmqU)hBS`&@3oF1c{u`Z*rb^%W>a=^LKu*<_qF7*UmZ0H8IllVB?+D0cN9aY)bbebSYI~4tmaN_jpHCe!PI!IO4&75|vCkL99"
        "2pSJJ$$?K>Q@=GT*w%De{PA7R4ZV|406934hMkp3u5{qwF8$cxHEeNCs`1;IxH(9q?qANKs|}N+m^GUeWztjj#N8ZN64|V-+>_62"
        "Yz~AK9Bl`-7QDs{;%3=+8iE=U(Wo{G@h#Mu2niozEeaJyEA$?+0Zmp`ya*u8E)xKOm0zoL_6DSfqId)Tp7Y*zh~Pt2FyPN9fe`^|"
        "tSgOB;B!ELp)wiN!x9<b7_$hTKeh~a6&N0%0fz+*A)R6^Ib7E;KX3vX*-b%msh&>)jLC57$heM-m=NsXCNX-6P5u7V+A1bK)m*vD"
        "GnH<20KVnkQlM$7Pv}3cP1dx<fl;bY;ZK?nYj^D!VFvyat#tt%(BDXApZRKW{HDKDM`R)lp{lM9sFa|wjeL39Y4ntWED^{Fn}Tn="
        "Ul&8~GiA>duE4=9W8e3G1Si{O$kD4{Q!Pbyn-whge);eZ*xk0<S~1w|y>(u$<zK**9`0JYj>>aNWf2#oJ*6Rkfq;vmtjJUp1(N3}"
        "rZ=T>!)@w)h4li&YgBNxFU8%`TbE@HZg$CF;jX%nLx>D)6yk3z1R%ZG@nF)U>~N(7)_J!gev8RV7H>HKN8(5g>AQgjBPEn-x>h29"
        "SRYg;6$csW9OY_Hn?g|+LD$yTK2|AE!OmvzW0qvXr>#tTjhW<T+8qQayXu{*bao4TVph>XMp>(>R#g7>C{(uP&<h?ocY3<vsTUFI"
        "of87N4>beN8{&-L96O=NDFZiz`r7xtw#=nUxDa8&k$+%dVoGUf&M-H=UY^Y7H(kyXSJAYsx7=V+uDjX!gRV17$y=Abbby?1%kC*$"
        "z`9w1COn(Z=ek*2xoM7|1OJ&#rCT@HsB-bdWCW(FZ8HRvC8RW=B{reAa=iS=3(O&BC^zi*>^!g>LIPTF`AN9p_eplk5iUT>rVbvU"
        "uIAo`N=O-U4z9oBkt%~`(&v#_v!aYE;u1WMKRzG72ON`w9mx<4qjY5F(gGm1{!psRvMv%M$5nH7tTkuyYJesa@hJy5g%;RypwN?u"
        "UP<W6YMp}?9&Ya-9w8F(TuSN?@<{%uyt4;N9l>4{^7u4iALo$D==j${!|!zLc;kyT^h2nJB1q^VlRiG+^Ae1}=XspK$6^Hp;`Gr#"
        "G<JLrGg=cqi5vK|*g<;w!_cEAxDtn+j7_~cA{hAx?FL%?G}&$jZX%5(cU)xY!Pf{AI@J?~<G;V?-w%Di!^f^;tjnZwkE);F$GlRG"
        "Qh#rVjf6#a*%VH=WStp|EO_#Npw<Ei=!kAa!d%Xj;95SBe{SpQP*yn*uKj<_U4dv<0an+JL#h(eC1{&>(@C87fa#<WVBDHSHDW-#"
        "%E^klXG6$om`<^=Dug*{cq=R7%A%)=Ehuj4-DO2G^-4UL!n75JT3%-K5C?FG_otP&_Ax>9F&#u+_T=o#+F#7)=clJIC;hxQJM$l2"
        "f)y<)K)G@!BiMQXaDjIX%8h`1)Fi)sXF&a%rFSo!0&MM}$fP$|8|v#7?*0L4VYp^Z@=cp6;HqKDfDAwJzF9fR#cx_F?$eaVuj>SF"
        "2r$}tGF&0u;Zb%n;;u1MGn^a!m5f3rZK9*U-ciq@yTz5Ar?0YJqN*Y3mm))=zY*yeG~x4S?>l|{cyx97(W~(+oTh@Z`2agO<t4z7"
        "$8>B6TRKmq$1vWioGZK&2F*it72m-|6}~hgz-Abt=#V(uizA6F7_tgV_ax|E)&|zero?hGOjU;|Vw?Z9$A_w}3^O@#jw>plf{esH"
        "EEJ$vkr+%J&1!X*b*ANr1cpSFcf0`Cwnt!JnZMvI@g3JE?pXA2y2P=G$BS-v3@!S<xoD1PwkDlEoxeTDi(ky5Y4LQ{G9{*l)lFNR"
        "(j_V8a{QiZbUYf|WC>(R9-b7{`l|t1@z?`(d3%v5a9EtG1o1(I44g)rk?K?94cZWR=45foUR;_JsD~FFnf8xzk)w0c+)e(iU339}"
        "nE6<7@Z{VMR={XTez)!sy50WmT+aS!&&FUcWI{<inF<Cw`OBu(#DmG3j}RU&e40DLNyp>x#(#L2+7i=VNtYX7tB#jGjMx;<d9*pn"
        "Us|!+!B4J}Zo{`s+_?t+RD&}iNHi8+bC8PDcNPeh<6mGjW^e8|C@AcQ8R)&M$3N0=t63YudZFgHy^UlFIUQ=NVe~cUp&EqWmwDo`"
        "H0Ahh3i+}QBvhBaoF+tu41z;GW`%BLHpq)#I-u<y`8Xu#$?RTxcTVw*3VTKGV2rJPo1NP&f;B%;?%S_N2tzb%M(R-ES0uipV?Fn-"
        "`z3{YXS6h`Zkt?6bE#*D^xLq%`GlZL7>!&m_yixz){))5=D^$Z+xNcg_SSCcx`{($ObPwg6J3$5=V}W8k>leo;jJj^3VB$k?@mE?"
        "NXt<a>Z})S@fa6--(lCQUq_Y_HC~BR+TSi=>)_}C`ryx4UwbZQlWAShKnP$u&$M3gG{CqxQtq#P*IoS^aK(WW_Xbc1MSaEJJOL;)"
        "V)DeiE|WIChDT}Te(HSoVRqGS@cePnK-bOIw-MW&KKJJFd9Qi7lLE*YODu<W=zO%~+=^M!w#lU&er#h0412oEZ*RR?HMDJAXLBv2"
        "?b2q3Tz@=hzHZPtArR`UZ2ENehsT~&$h8OBoi7sG?;oDs%_x6C&l|Yz%TfsBHb@}a%(;~*=y|&9lJ1e17Djua+fM6s-@B=Bhpz?C"
        "9$$0&>4vs*!}vc?O9KQH000080000X0M+peq?8)~00CqG04e|g0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-YWpHnEX>@OLd0%61a&UEX"
        "E^v9pJZp2?Mv~vRD*pk8y$_&GLXz!lWtA{{UYp)ktz)|+IhB;GMS;MOL<9l|9+DWZ|NC|KJTNmDP*hS^r@M^_pr@y&r>9@jGrsSC"
        "$clocUd!%V?;ly~#dWgDZdv2SMe5~Q!it8aQ!i_pj?Ky<-+48w$|h^edgoodx%A?63m>zx2z}rG>Q7(&Dce?M4Tzdn{ocgQCeN1r"
        "bK6#7l4q>wVSZ`KLOs+>{cPiUmF4|&SLc8#WOZHE_RE_6+A-{RRhL^YDf65qEs)lD0`67W6)md;KpMAklE+QM6aWk^j}s;Us<=hE"
        "<;M^3IQ2esHTwV@xW`8vFo7<!Rhh*3nr*jnn3P3Z$4RTuo-3Zcfw4@r=Zhk)noZfd&>($b($z7JL81!2`jZEL9~S37oLycnUPrIq"
        "zg|Rd&t5IwT`Vr9{OhYSW$$lUoyXN#k+OU7I)T?wldX%mg+Yjy>>lw(RaxfZ4e~omvo)|%ylJXDYa<v|0SsknRI{Y4(+0jyMhP!h"
        "t2`^9zf~(~&&xQCXh492I`o=0UP5oHtXQ+Us<UF86EGXerk|>~s`6b#FQc}M8o|h^$6w*xHGn@}u?R($UX0?_gxchD1z+Ntqu>By"
        "@PacngCR9#S0?~d-Zknq^q8d};90xV-U$?K+}uP+3bx@~#Q!5<^j`8MC?(?pTCh`&0+!f0bcieUpWTw?6M5>zzz%rYZnO4VuV66q"
        "Fs&qlUWq?h(zQ%qx!6)JSVWw%l@~=>0ka&5k<+}hd^PPK>#}UM$0yqFIM2(ws9^0~S>Mdx#CgNCuiN-O5}GAi?poH&PmYhHWBxx2"
        "PEMo%{CW%yr4&D_50g`q@DR!M=Co`B{5^qbZp!?Y1rrO>pdOkS&lc=pEC4@v=V)V&OEfZHI$F^v1u*^AXH2#ZXr4TjDxC>Z{-C)b"
        "j{yRb$ZLGy>m5LavJRU~{Os$$1cAcx)bszxhyPORW!$icOWY)61<F$}nS>j5&k^Y?#+3(%`#CxC0?{4d3_-((>)(8>LALS=0K;}6"
        "yU&`o3G{JkAj8?S;7DxatYIE%-gjmD224P@K<g1m<0siDjUQM$kBnNe4t25&!3wuGX;ufQ9E!G?U)CMybs{z@Z|Ko#*)mitagEbK"
        "BfHDmP1JO&Rdyfvpxx>=i*vib6|CR5#kQbD3m6LM$>6!o4Z>gV9B=UtJF-k<2#;mOioo51diK#wH>XEOQq6>GK-4Y4ylsy@OR@NJ"
        "#OLqmvn>~2{OLIMHrukYYv(V%9s!zw2LWQ+RA9c@$iN6u*~U4r^mrVE#;@2P(zyM*rB7XKz|qO~bwC(4AUs~O>53cgbgsw>QTqoc"
        "$Io7PPdxm`H7{<0pU%A;$RL!HhZ}*o^LF2lD4_#7#y`FkevI>ui8Uaxq~%>f_IcT*Yt}~WK4C0nsV$JMKhDGOuFhJvcSuWM*v;@0"
        "_Ejm9JHXz1i{U>nFFz~@8eF0g$<aN;@Z<Sg8t4e@B^SeeAg{RB1O1IbksMKXBeMXrC)e<N?mgQxSHt;H7D7!B*!7|zpKTRqIXH44"
        "$mG@tpIFD&OTF-H15`bxswHIvei-O%_y_#M-`JsI_Q%AjciEB>T$Xsv$Wq}D!F!P%TT+44T619eS!(Q-KC(XJ*tm-^Vxk@J+~+Wj"
        "vj(P(O^zt+DsV7aAo;?r>RfZN`HW0G(?4=diiW*r;J_CS268_VeS;w%iF4!$;D5;sfEzA&1J0br5zXncd@ZTrq+fD%&7Unb+%Q&<"
        "$ybg*bCXpS7~8oF!sG|xrM)W)CZiOj$auvyIFAIm{^o=6>;ItVfa_0;QWH=#om;kDf-?-3bNHQ0Q{Cb)jG!*yt->PSvI!aH#u7N`"
        "pI~Fv3&G=@!CNFa@Q;8r`0R(Fk0cQMsDI*GML5ccHA!4CTMq*b#5wR)kWVhOoUzpF)$#QZKrx#9nVN#Uz@Pb3?__c{JGl<Qt?K8="
        "!GM~Dt1i#C7;pvt)$#1F@oaVd`Nc7qeu14BfUx{gXOCM8+LAd=#F()iXD7#~{9p%5C9}N$;#_&uY;C)|&8i#&!Ln@}*oB;A4SJ3)"
        "@-cCc6&-Vmn`(`#z}Z}^u`3?}@0%tn>!>J;nytG$uKjEDIZhtYXHLru_%ZRma~iEZoZ#Lwgvxo`N<p?6q8uSHFw}7aBBa-&bQ8)G"
        "Qbr1B;(f1$MhB6obUem_6t+L$@W?U1I)Mcp{!(UwEmtR}*Y=Q{KDe2K2qZ$Qt#E>RTK7$GS2SIPJ^&_CkhhPXT}eflfQ4L@c^XVm"
        "N1@_=93K;IVZrB0amp$0GK`v75yoj6IC`}~Td-VPjKR4qg(8tbLeC||YJmy7x|Iz<AL3%^ZHLYYSodg^2$DPRn(RR^dNN$I<fZeb"
        "6VwV?gU2liYDb0~mj~(0!HRH-59Ma(lKwDd6m0ulyPADtV-Q00vB(hUHGw`TOrRtp!1m4Kg~<POOqLBv+}Fc0E5HYRBz{`cVGJ0G"
        "G%z!$N{*(~wUi;DD1}`UhR0S7!zn_?e8+tyDb<p%!qFP$YXj~_N$;ucK<^8k1iswozB|7v^DNo<r=E{;d;<&i<K>%gBB7Rfx0qOu"
        "(A&$B=;F<#9yt31K$rnXulWoRftVB|K#|ha0mMYWr!-62Ks*{aR$W0sPMk-{h9x%;6`{R^VYLj}=^4E*x|f-h>jGt3>=gtwAEz>H"
        "oZgd5A2tlvK~Vb_A71}7d&^;GXJm+18LLk}HBW-8e}5{jpH6%+scKFUjW&U|ICZ#;;dQvK%dQGe&`GYd3Y_JEyC9~Jw*qY9M=_Q{"
        "_ZW=S0}XpqKn&w^M1&RkM8eE20ArxfHmk*c2x_7YtJ!krts!D-aT{e-Gv<{mFWJQPn$glBWLh4zNMSxV%xlV3dl%P~jH>GLHbc{7"
        "HPC$kF-#xC(N~oDaTUXC%8i`9J?{r%2BP#?c-((-;N)0l2<9Ag&C(3{zwBtI0&#Z@@tB)fdc$=9dga?@&&Xl8zjNr8L+LM@boP9f"
        "<T23V4^&s+7Yw{mU<Gm;U3ZN;=(K~@DQg%7{01VnWvc<$T9+gS`+GL*??E|fOXL>nl4=LxR+(li_!ro51YIk}GSA@6C<n79IYdkl"
        "tQQkP4gXK_Nz+%bBI(d#VSZH32WVXeb~^?9Ks55L>Xsl|URrMBtk^3<T_J_g(5%`z>vakx=~JaraB`;7{!O_;mt*pM|4oVcUmzQ|"
        "vY<_dnbMiI*>QexDOUh|N%AgbQ;+ZBfXCuxxKAfS0=_3gY~kBLh~&N!JAL?m7};W+8THB8Z;TZDibQL~vl^NrL1rY#jFMfF8=;@_"
        "q$oJm*(?-sN($H3%F=@|DAU@2W20umCX!DSpM<{2)ZQqN{>cNI0hkLbcGg<m#`O&k(<K&X)JGA$0(AtCKCINL&RC1yh9gtwuZzxG"
        "vx3zSR0!FXVmSs5!Uq23fvq%x?EuZneTo@wxP4q@MH=Th7|4Hr3O|Kj@V8R|@E-`uv19Q#IR#VsH28G!bOI*xzkO|80Gc06Mx(bN"
        "`yuD`WIs&_Ed31ARhTxAf*%&3u_<r?O%#}k5|trA$NET(2?A){t+(PI)r(1N-#L11Fu$&t&LY^*4Co_E970Q6sTbFV*5U1x#6MP+"
        "rKY0vj9jt|?NpJlz}Si2=$h&kI#x;*EpaC*25D_bu!X4<`_>>23>{yfBHy9Qu!dmC6I<ZDmXc|>&y*R%wx{$ilsrMq_<}T$AS|R-"
        "UpNLywwBAoHZ3|REzep3LE1EkFLQfZ`Fm|nfwKU`y?FQ{0D$e*R^O;kNcIG<STdmIl=6=XEdWlmBp2G`y$!8|m=Iz+6(~j}WZZ1R"
        "cZG1P&k_4za7!+u?~@PP%Dut~?g99r&%Fly{ZmXZ3^C>Lju2hB(_N-{9NBomDK!_1Two!nmu=(9uZiigvzmNhBk<&c9QkqyW7{i="
        "S$HGw)RBgi?Hs`4$v+~l&;u>|&)=iiZJ4mVZx4dorU!x5%|Ym9=&T<C#~tuPQ64%Pf%ZXgbWFifIs|4Pt}nya4rxHMg#tg4+%?v4"
        "LkX~e;2*c+=v<K_U5=no3WNU`Xb72%Py|2d{K>4T$V*bSmK5SKSMTaDZ`rK?dfKiHy3s+CR!Xeov0#~;->I}|Q{f0J3T-FIkw+X3"
        "HiAAQtSf9=GaV~!l}rkmxE;E{r`+hNuvsn-Pjrb{lM(&;St1cmgPf7me$YO58z{YkK7vEi@vol))l!tf7(8<<6N_>UznQ0!3^H7l"
        "<RKp=*~n=UXf#fI_Brn&!34AxSDYi)Q-gi{i+XXb5s3rG&(fWEkRF~z4mfRMh}Zp~NQCd(eP|jcYCkpi>qk~YXzgKn+>Sd%ang3+"
        "cJDlBbVrdm2DRMltB-K+7)uzxg+<_%bqt{x_lC46an<)kPL7X$S_JVt2(HXUQ|@eU2GSl7Zh~d7vLnHKjLRd{kz5i)ORok$-+|Lu"
        "-pSb^uJczs{4~hq9~y9dudXLw1j6_>&T_2BG(5_OeMqfdtC49}43N41>wP!xF+Yk8N+A;MOcq*N$=SoaaJ|B^Pv?H1@=hwSeaq6d"
        "GCMHAue+L9QL%bfc5PBZ7@w4N)ioX^NUJg{+U8p?Eg7HY1`9PYf5UPtYi-91EP1&e-@*nywJ^rU$Qtn6L-Q-fe$OQeIpwmX(W>O2"
        "SWVcX$JC}#HAA{YIpY0=qR!XuKmdlKoHX!XoiAQqF1*W^e}B91&fa+M-d}o)pUy5WFVv_4Rc-8DE`GY4_<QhK>AOAH#dI`5xmrO_"
        "-r2j$#XlD3-iPzEA6}mS?EP!;^R$Y0R2s{>9Vkp9K+!gL)FB6IX0#SK#i^`i)Om5Sc)NIYsiAyx{{9EW27eN+80fOHU{)?fU_}3y"
        "{}h|g3ekfGv=?Sktjx>BW!IQ%hB2j~qrj5)X!S4{+8G%7^Gjs%`yT6$FW-Iy^B?>T)Aju<PbZEE=9?ruaF%NUYo%!&=yH#WhEFN3"
        "4nBWeb?p-r!Fth9wj=&aLGj+7zXtjE`_BSOmj`S3yNp4gEvdj$2_fP`V?>CJQ8e)_6RX*xdr*u##!*G0;^>s~)F2`4wpCBUp-fKl"
        "vS9-*s}}TCOJ^**EKl`HcdqUPzq#&}wd^o`v?(Qz3{Wlu1up{4BVM-U4MUHnjuMPB<XM8YOA&Rk%GTl)@Sy6Xd#1z63Ugrw6RaG<"
        "(s5Pu4a%9PAo2`GKmS_3?=1<wesTQQzf2`t$hM0Q&yHVwBR^!#jVuLNXG>@!K8#nkS9Q4tBi+n#vl*-SwVX5<?BBq@@MM+2jIuqj"
        "(m7AR6<!FtPx4Og7R4ed4NyWVhUw@$ZgB@ia2>1u_!mrGZCM-RX$A=_ws)jNIM1oTBuMbZPvGRA{jhj<8C`r>yt)``MF%Ft+!{wy"
        "sNU}uovaLOwVM4gJT2(50%Jx^;<Upevi<Prg&Yv7Po6XrLCR;pxkDX(&uSmN@;hw21b1K)-9GcbJm7oOOh#U=_yfy#vLetWacYU*"
        ")D*j^DQr_)#IUU9WG_uwaU17ZD%PQBDhYz9t6z0sk_Lxs0&`I+r)iB_N?3*7f4}l^6m{!gU-`r@%(6$nb}?>Tf(Pu2_;8t*xX~>3"
        "KO`z>!{7H)m0<)REC2Ft-3*!}b=fq?9tLC7=a|4%V;Fzj@L*rl8=T%qe3aC2H6q<3{?yMNe|DWG=+6f%=%X^Ms}9{y3=6QllFFOS"
        "Lo_%o%CM5+lD*S|{^4Ms?9Uz^@!@Cfu+$MLfIyvc2|0`Ter23v^~!E0cj!}H7BF{wAButFZz{SpPlV}*d|)u~n%8K}iFiwJ`UJl("
        "PoqVImo-O9TAGl%k9zipYEW??e7O9BlnU~e=g!kkoO{@xJdOy<+0?_vliY~Q>IUn&R_IlOprAaipbJ~Efr}+F`}K7A@kD(T63EMo"
        "6mq@7i2H4QUxVT!)s6TOAvUTw9Z{BnHVB@8HjSt>{UvT)M+&|1_;4pZ$@_Z|3ar;;4`S{o-^F!>&)KOVAEaTMVYzUeV1*OUeTj9i"
        "jHsU1Cj~Kv#8I9s9oxB4ydwh6Q`7s=K9Lg?sr-*#*R6acv(BbpTUxMzO*L(-f6<dazHWY0UQ1Xb4a~Gl0B9L14b(7M0BM>q4bU)V"
        "nCdc&n+|B2Im(I|Z**XFd^i%`3KmLf)<Ble-9C@3>_B5wN$J+XVyr8))1b{tDkYL!(jP`;Mh{o!5_B(2dpT*zSA&=uW;NTQ^jWD*"
        "zuL7MVwLm<L;glGF_>5xL!S<fAS3`lI|*mLAGVV0Fx&KR^%h_H@-X}-r+f;mr#;ygjeAPv^LCAorv&#!<#cvX*qlk9M=_L09;12X"
        "NrJgIJCVUkA7;wT4py#E$00?gx_e+E;XgFGNyS|G>^=tn5d1~E$dS7q<jqs&4Ks>hZhbrORFxe9`bt4wn~}TEy<omO%3%j38wh~J"
        "N>rkWwcjFjsm!Nys6qX+QlwWL0GzkDD2ptSjvKg*$JegnPRShE(SSLQUD{roOS04`)F<CAyp|o+>=()up5j4OS33lW8~hcD^A4x}"
        "#G)S00f^9!UxT0t!1r_SSR+`%1)qH*Oli6;6HRcVpho~<iDCvX=ai2s3;Sp9g2bK<FnJgc*63KI3}QTn?mb<?zyn1}O(%JrZLvlG"
        ")^dv17L-+&i!4wK{vpO#L&>jW$_ujsNJXrHeUx1@U{*<!;Kg@yePs4#`=F**6r`x}A}@6V_CRgQL+ebgWH8{`)<dSs33%am0K?|%"
        "Fe^a26fo8-^=0}-Ppk}!v1vZ$qv!Yo;UgetZSj=rW&9}iRE9dk5_E#e<6_<65jhp>3L`K@b6^Cw@^2VWVx_leHteP+641*?4+KWN"
        "S7!+QBkXnN^AiIYuN~BTA6XI)wD8h#wn@`Eihyer8Pk<HsAJhg%-XMA*~WuD`@`}Ms}m?1ro(fE{7t`4*6&$q15)@ObI2m?u%HnZ"
        "XG9mUL~=JzSdb9|BD=D_5A6z!Rl-`RTqiRrVhh!s19aPf+05RA;V0eq?>`QCXWJ;;qEk*^zuh(D-)CbBQtW`i@M>cEfH-|ulAuO*"
        "FY(VlsG&3gtp|Kmo4Rw~d-?9QCr%1k<lBOQ*ME2s@@edz5yv>GyWwla)8Qu>Cl0?DnVY$Ia=;1YG#CFmAn+KLbkWH}Z02LJgo~rj"
        "qc68vli^Zy2-X36z%_1$O1OUa)b<?)2e0-W=N{j*M2`av47Hq$AIZk_B>H9}_t!jPX)M_ala_rApez71tN%u~Z4HVJLPL;l99Slg"
        "o0EKtUv1$J$Kb4$tLa7>s_NB8MV49|^+hLNqtyH~b`})P8X`tFj;F==VO4y&@3Mw{#*0c-N}fw|-HsrS)Pw6P&H+PohKH1kRIx}K"
        "$OskV*LmO{xLuf)Yw*mG!gPh$ZRwVO@{WazUD3w({3-)&fOscW(ouj%&kx>d;O3IdSj!pbQE5Yc?zyI*iZeQ0Gn(>uXhD8+lKs|^"
        "b$XOOlo_hgHe%;!pK8<a2<Wj~3I5quni2*%<)_eHHWCQ3AX(RJScd~vKWZwLWMFE9?xl?K&<*L)e{ZIG4$^gIHUrndDlhNsMIh^D"
        "IATLr<MUIKIq>}T$ul>G-y;)45qJR{qAj{%K9f(6N$}q3fga?Q4h~~Vs%vh|j>DL)T2{|}SD_>&;q_Qj;&H%QQTSYQlwFpd>s7T?"
        "KR0iKvt-In1>s+R<X1R_bNRys?5OLLHhONqa1Bn{?R|L0`~%m5jKLH!<uA<Xpq#dah$W!??x8r+l_ay}ZuTNPcClYjcEfh_L3UXF"
        "k@VtnTZ1v2k5nuBVnO+*=tKql+<hh&dkw*2Gnw`)*EKbS1{*~}vPE)%@JL|?9}`AL=wQ*Q&q7zz>zQ{lhMj}yc(sOOjnv9A$&#;K"
        "q`5IYHQQ-jyW!mf`Qj2ZULV5~`Mkn_k8qz8V-I)`v{;F?(#{ABGIMkd$K819O$&m5N1hG;w6vB(*{e*Jo1gm%?FSVjuKkihD^9aK"
        "lFXmR<cJzju1QOt^{Ji13(oaoV&|n(nnz@Q_!rx`UJq`Sv%Vj48P%RksN|hDmvL=>9`(?8f>Wfscr<cAvXr+PPl{@66BR2ED)?Q+"
        "FH^w$7fgxDaq7u6W2QWre?1jFei=cpvcKxh*b36JyFinA(h~SQu}{TXR^H#ttXs0sblae^Z^oiPhXR%1CiNs-=jAf+pI~MTR$G|y"
        "{ORD5i|_3%G4F}@?8%eo+9h^&bRs9MX-ho2A881jtKB|s__*Y&-J7xR_&|4p-KxBdx~JI7ctzzu=!xA<`TeqVq~CeZY)>|C%8OMF"
        "->TK>A|q9jd8)RI{@-_Xoh3J?s#M3hU+UNuJVc}Rs|5l32OCUo97O%~6>6>1mPAC?SE^(z<y51%kaI?pCzy#sq&rZ$73Q0Xjbk+A"
        "i#tB>Q7Dh;p+L;`iuQ$Kd@_bE%K`Dr_z4YTv?Smtdibw(g$EVGMO7ZV<`#IIY>*qaTLbi>qk_80^%Yi5w-viV@co7ToyRgD(OsTL"
        ")CoY53rZBBqxgK{9UTo@HXTepIN*$KB+QCP%_Qp0`pP$aU)*^U9wg7p(h+Xn$CzkXA%Jx9dx<Pq3_{lD00D2rVyyNPaRDvd+Y_@="
        "Kt?e=o@~4A#FiW0wIlR7?XC6KqRe1z=9s|eRWAu=>F+4curHo^uttLJ72{(;PgG;^(02vye(#<199Ejs+?<!^h5Ea{`FxXb-%e#k"
        "uF$mlk>_Ea+NcuNt9OmE-al3$EDM1-=jL{4r;g4j(fxyk!+L%2p@qZGR2Spr!Sdo^9(vNz0Ivc~?BO!>_(eb}+&&UC9mykk)|LCj"
        "T}-TPy7d}V$SMZe+H~sL`ppWT)Qz0u<-j91p<lfJc>Zb;oiBboyEuFQ?&ANX{Ptsip_yJ{$vX)96F)0fiwyvqtSb&H&#=Rapqn@5"
        "#<eRz(~Jn5k+{!JPL6Lvtgz3uBLQPc0WEf*>;UrXh~q)YI-Zp)`f-A(*<U+#%s@M?h4*Wl)1gtPMrS51<xq^mO4M*=G4Gq4TpQm("
        "vfEt+D@EizbVWsL4Oih@D^*2DcC*sK_-dm6nx&gUdF($f(63evrFD6$Qzwk;g%RC=e}<~TKWk*e5LNF@R}v#y+tL@K^>n8G7;{bz"
        "%MRO`W!yxVwXs^UEs-wuko?il=KlgvO9KQH000080000X0RI=g%=ZWY0Hz!O03iSX0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-ZcV%;J"
        "X>@2VaCyyHTW{Sq5`H(}|G?G0Nb7UtwtGS0;%s*pNxB6V-6C<@hr~UgCCZ15M5-he`!vXZ-<csLQI{MiT@<Jv5{sANa4z3`Lnle{"
        "<deU!S{J-#g;vh+!m&5JmL+#mt1p<+j&b(lho@WHNOxeRzE`#8r7%nz=7e?Zz>N~t=I3YUXXC<_yW+s4Ww+hkR;scqq`J%5^#O)T"
        "QOjFlxD)k}8IWOQ;Y10CnFh}O1%w~CDr;fS&%V2Uw*8znx|EeHNGZ#7#l(ZO4kWTl!m<s6<?M|xut>m>dQhO0MihS6J*xGUbf9RG"
        "B&e-yT5TMxsVY!t_&#uZh`)cbS`8mY48Pj$wl%tdyAMN7G%ad$epVUXFt8#oYHlqk)gRUmg(aXS9@bBP3!kVO&E3ihJ##0ThUdL|"
        "21(CF%cXhxz?-%f8~oH(I&F^6<n;T6;;lVshjP!)N>MQzM)r8swEwzV*^{yD-x)aQ>imqsPs@+!eYRsS(<(>ZH>@Jngo3pd#6-oU"
        "Vipt><sdl^UV1H#yP7w*C1=$Y^8&IPFIrZt7Ak$)l9YDC_9!SE_%36A*wK6#$bv%nW4N?}eb4JoJT*p}G<njsHAagVZc28xd6M~U"
        "=moe0dNwb+rnSA?4HL?CM(nvQq})B@wFO&RFg1j$Hg=aLn`Fb1t0c>Ep$c7!H0fNm{k#`#gzJpzgDAHL{@Gvumg0d~eh?3(yn}#G"
        "Gw+Ol4B?Q1$A1xOcWpWWlQf2T_X4x&xhS+L!<D77j;liKWqF0J8E<5T+Z*vlT8Q^(3NrBS2ou#}Y={G|Wi)>(CW0E)c48`VPdOLH"
        "KHz%a^!Z+{UR3zdzM4k*Z*tvAK8W(f{BXJDlp*EbH#=e}4MNN%ex=c!DgpS~y;k*c4>-Hm6+oQi^&Vl>CS$tp@1)!Fwylq;$1`xo"
        "6+y*@jV9l)KW#u8KtT&OyVbge(?S3j2T{D94)z&TXh0W5DV@MM7~I<e{{>HooOezarml7s?!jTwgH~<t4-!;uc_DU6-$NNl#$4_d"
        "8PhZzl!E}F$`tm8HGSs5F0bdQJ+xxXHQUt=R3D)@H=Rls0@wxIZkya}AbZ-*?Z|5L`~@@Jm%G8j@X1?;SfD$Hn=p1a%*jT;!0ykV"
        "_*1gz%BW550RSbR#HeqnKKYvE3mdCAz$7Gf0_p}~)<LN?c`0H|!T1n$HP{_cXbh8~eD3YNpehVmfccdA|E^|uLIGm|DILWRLH#2s"
        "35D~ycN(TdsIj0q?~894$OI+afFfYEgHV3WF=Pz!1(f+jLv=)+AyoyH9}?FxDt|rDv9F&%?GO)uLxKs&s1Bi+$OnwWZ{k4|9e_RB"
        "DEC6Y0~BvtszOjk6*tFAUY3`@JcJXMMkwA0uRa`(b_WlEjvy772eaS@<$#<FCe4TY5AH^ly<zh?8lh6r^nggSae~ks=}((JQqcSX"
        "tUrLv0VPhlU}+SNaZVx`Hv&d}?jV%QJc%Qe&K=6~EePFXihvxbuXA-zs?=8l77J-QAcV36<DsL$ppkk|%^kA8Dcg+Io6?e*ve`nz"
        "Ib5I8_Ft2Dr0UEXX&b_UNQJ4_JgS3-xN#QwW11jYWZNwuB}-2G?CC=*h}Lpsovc$nKTyDM;&25bNC0GSJt?{xS=C6Y=H8<{?}DJ7"
        "<)V#pO6t@k>6cBGe!2TGQaVdEiwg#23JCo>L6sojI367&qXKZC6DIveE8;}{#MlT2L7f5VK&Ay9qO<vB`trwD_U0mEx5KZmR5A$5"
        "Mp3~RK_ckke|!1q4aPzE`0OS+eBBHGh3X2n|GCrXioR{wqU~l~1t!f18Wg4VaqtxWpl8@_)MT8QlJC?L<0P5hhpQvZ7&nP0IKUHL"
        "7(yFkRj2(~jzbgkuHJAuO(6AeSwaf3MmSDHm$)z84^0{Yqfb$vkU)X3*fH4Bf|v7TDtLMovV_O%c?{ECU>0UVKt1XsUkYj06hI)*"
        "#!hcYUh*A~!vYanLxIR=hOUxVYP;P+yuC%C?_Q~kiwg+3{wo<&_hI*_g8!VCj!R!th6)nE7E&E(2=uy?F@q(SVK6<gmP*7yktVt>"
        "F**gqTCc?Y_z_B%Cur@Fe);O_>rmu~55%G|v`~C>TaOMnXwzu*qk-mA$%zN|(TNiRp{H>a-V?}OM#>7FHmwU5AB0m+-#6X)wZP68"
        "%<!b=aA^ybM?UP5xRu_Lj4dP{x3rn|L}w2a5R#qpaaa`H>5uFO1N4_6eGf#4^6U$ykRh>qnzgz!z+i~A0&9%f3h@^J6VsE6-e0HF"
        "2g2k|Am<&myreB_%Me2#m-6N{tWN!h4Mdg_Yp-ATJ=w}`y2Bn_o=&s;B_4Q#%`$K<NY35BOd(h?w00BlTWesJWoYPLf`VkUxpu!s"
        "RJ7eUdVRGmWih{5wAS;~s2qNngD57`xp%T$L+qy~4TkXuY*6s#YKiKR=HB|)_RwPJICz=PJQxyd&27;4l%e_n7G^yuT^ul5x1!TB"
        "N&geqk`WifCAR>F4FDD5gn!rz`9Y^nAY*huj~dmKN_a4jQGI<k1TXGhDj%Nx@<7b?_`R|We^#v+!RTutP(`0v*=AkH(!L)p0lFgG"
        "zpIk|{>gimIvp0?o>(|D*dp|OF5#O+lK-ToO8xKz1i$w^9rB1fq3BV!>@$Z-8a>itJBLf4W_OG3@T1^wC9!^Mlp0O(k(e}Y%zg&|"
        "f(=%T7s#U*F}bpj;$e~K{!I>&s2qAGXx9x;A8bv1<3rL%H~*T7@PE{R->c2nV)TQ}4t?xp$S$9p!#m=#o%QD5t-Ro-ai&k3N)WCi"
        "s7RpGk#9t~C1Ej$xUPjxXSMN}{5@bY6#Z==3cVciw*fMm^}siMN`K#*RCJ5U3voKH1OBmPd^Vp!VS<0Px&<d6E{DHg&Eb-udb28c"
        "c;s3?79V65QxT*@41BY2%lu(!xng*%dEeCRl_Xj=eQqIt#461Gg0XlNnZ2G){<?8W%y{<?5dop_KTt~p1QY-O00;m803iU+WD>Kr"
        "2LJ%o6#xJw0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FLZBfWo~JDVRUJ4ZZ2?nrC3XI+_(|Gt4jX?3tu=JijusW&1URz)jsT$4{^3^"
        "RW7oMg&{!=ED~S<(30Zlzo#1y66B1eN<|lmM0cYfUw_>crS$i+AJ*h6*P;=;CatPiORT!#Qv921!i?e8<U}=%s019nzaW-(eakG*"
        "NzZJxJ5jQ|Czk1JkSr*r$4AFUqU)8mWXH^|72EOoi&66UsL`q;{tX-!nj1Cfikt9`G@|7uh`|+TAs+n!k7(bWS<R$rl<v45b$=PG"
        "x`^(Nl3USWVAs6sSWzk2i0i2E+m2T^y%N%X<V{ZA-t~}+j!XNg=hY+XaB*eFD?jp{3H|ntL0TXw@%DQ59$h&;s(C}uVP;D%D^+t{"
        "qB(wNGw$Wqb8{<l@`f0z*T+W$el(Y$19jRE`dNOKwD<y^XZ{d>8v5eJ3-V5&p>LSS0u%V~S^5WenY0fy(NNkYoCrw_gv0CTN?|1H"
        "%}&|E6(lD&{C?B2Zd)^_u_ogMD{pTVy|jat!I$^O3>@x{Mg=}Up!45iG!JyjPODg5MmUxu5VLA5(|TF5QK+mKh^-w+Rv}}6$KUWR"
        "(Mnl@@iQR}(d_p0TB+-nm!&J~$&V*RQM~e<a8I`P7Vx;`kR9%P67sr0AJOJV{#6*EBoQE@23dm(#5u_86G)?L8i#$%nuL3WY`Dg{"
        "IXKlq^U5lHPbBMj;f;Ra_EzZ|(s1Z0xQ#Hdt9hp^Px=utLm2oZIe1FAWZRb4#TcvqqhnGu+*p7Kc3go0m=q;g&y3Ds?K_>W9HOwM"
        "8J!Xv4Hjq)WHg5=dDtS1!QBlVY;*E6og;xmH&xc*HJZ4bDp_<8O<Rik%1eu8RJf!%B$-~3?=~b+iU;c{9?gW|<e#h^_*<=&&ge)1"
        "L?^|<QuIJK&;kf|3$S2AEzrR+!Wh9+q|Fjx<djeV5QX0df&mAC9xI;FQ<@W6meC@18WubU+t*kfaAAtq)r4LZ;7w<;RT3Q*1Du!~"
        "I(23y0mz$+4CJ7_O0+aHFr^5I;e)d85VSs14gEWGB;pMyz7>)$$sc&cQAms;To{<WFgy10&p&1EZifojIoWDe?D(CJGh1N={oe<@"
        "N($E`mBTAYCrpP^X6eS{<z~+L1-4iTRR)rVaOwp1V+M0S=Uk>tiSui7p$908)q}C6@46CuZS#(`h96?L(Sdv{;>2R{5KJMNmx27U"
        "*cFGUKKF+PY^@h#C||^!5?wg^K98+`H;xu3Z`Ob_WP%^h{`ylQRP!f<+}Jrh$R=Km=j7A7ix`u6MOx7b7>=5Mfus-xG}W}~7VH}f"
        "_yCv@18J8Qmj*q^irG20&i0S@zZ*|?j=Vd7fB?cHw*p!M+gHMiuYg%0@5xuF>2_!tbO>B>{_^KQix+M+IcYTK<eE#aK?nE%s_aI="
        "`?GUT(u~wY*P&PN9y`MCG-L-eYRRyn<<7|K0~sf;H{|?>^RvIaB-rwa+&t-%-1A%()7k+*VJ^800Dqhpg|{&BR`4v2TOUK|Hnr%@"
        "p|b*(4AIYMW8V`jbf`_t;lBoGDyR_{marGvxC#2*aAH#*cpm>KlM(gU$j447M#TmAgWa$_ft9D%syL6rsi+O8WQ(_nPahzfF`v$n"
        "Ae7`MyvYygKN-`gcsQngN9};*5#9y$EJT<SR@Wukh4nlD%dy#NPS}aX%QH7qqrI&%UJuTm<7fIyn3r2-(os>1I!@OfwmS0bKRy-d"
        "#OQ&Pdsim<(etPKYG7oL{==gd4bbL};r4$AfC6;Gf{nW(8QgfnZVw*Esuz1}W#~>0oW1_PF6aL4!tJTyx@f)Q#xNLWVDR!A)vD!m"
        "P3T~_#(5uBY<T6beF)19WCo1m9QTMWSl_qzWd*mE@rx>%Rq>JuBz_e6a!)i?`dh;M^7~+k)Nq4K+5YrG5^Xu`l>e(E<WFg|uo?(s"
        "8N@jY`4F=_T;-FsMKIIB;i#5U2jk0U0S5kyaRRR~Z4J_8=1cH2#U<ajq*grH88yzC-E4$-gCo%W<E&?6jziCSiGwZcLWw>5f?U8&"
        "6$4Tg?t?tK1hs<pa(v5g-MYpa3mt)uRju-!0the2-$pm1Nn=$Hn$`(VJIpJH(?#j#9$0p76T7I_qha^01LHLe!)9pPr-B_zpu=o="
        "K5i&pw0gNFuq9+uPFIe!rspTi`0F2ULmZV>u*UkH_R6PWLXF{!JEcdh4{7vc;RRV=&489D!FZjXE>F&#!!aq)Brp$b2pfK(P)1*4"
        "A^!7*K8s3#V6P;+cN;(h8wza4p_mg;QVjS6q>I56Mi>FnxkpC8)9v9LF$1#4jPKzh%vmyUR^-j0H3{;Cf-%!kYO$F<zZz|a8tcjL"
        "D$Z+Ozk<5p|7KcI2|MnX{tFPcYWEMLdh|A|^M{w+ga^k#Ba!zi4j#{lX$&?`2T#m*N`WIp@|0yQtC?k+Z}S06qv7y*zTR6l@`Xao"
        "-l(wLxRweFP6dr6>pC&3RBIkR^hxIRV*+7>mc4?i+`9?V#_uoxUnlr*#Cfr!`~IbHpzA$wJ(hu)!U(dZbg0oxIBEC2NPYEY;!{QU"
        "#P?ATI{PmVsqKZa|1wc~n5N)=0Z>Z=1QY-O00;m803iUS1~;4O0002n0RR9e0001Ua$_%IX=ZO@VQgn*ZEayMWp-g~bzyXAZ*DJN"
        "UukY>bYEXCaCwc7%Sr<=6o&T$^A3ls6r?X8F03A<(2g>7A&7)bdZq?*<4h`jd=p0t))AUj$bZTA3n5zSiOFk%K?f#aih+~yTj&GB"
        "-1I>!dr!`(OX#xi%_)8d+LPWT6Fis!d`MtiwB*Pq+RTIybyZhA2M66)uq_2M`cQVKO6=<@b=Xiz<`3blt{@+Le#CaM!sbb?9&T~D"
        "z8K$Iyu%io8<g|wH7*wo{wR@+Pr1rP#byXzCe>N`CFV`a@JSxpM`BZk96aY`jFie+Dd7rU#+3L&D!%)~1nWVYrahAX;hlCJ6BpNO"
        "U3~&jO9KQH000080000X09?f29bpFm0Q46C04M+e0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFJ*RNY+qw<WMwXJdDU3^Z{s!+"
        "{yjkchk=SBcHzfn;TApTT+|Qk)_`BIz1ss0g+NQRO+*r1l5*na^1t7ZlJ#_Uvx^?uOAy2sheOUcGv7zbvaBUNaG^j(hl9_k2Lra{"
        "F7e;fARr)JE~o_J^Kd7bM?um95kyh{1d>O>#R-CLsi2`tW$XIt`f9yr5(4f;J_V2C<d7?%kJOD6g<qn8g!Ym64VVrViw5Y=3qO(#"
        "-~v4zxMYfplm1AU07c}zcDsl+K9hS~?YtvF!R4W%j|zC`pYF^z!F?ZXTpr*qndG5FN54l5jRh4^7_xA8eYGKy`Yc2*4@~Vr?J+D8"
        ";BE;nM^yv%MBpgy2_GXU;jRHB^g!`@8nzYZK5QayM-^_&vMe3+R`9^gcYPuydS-z^HGPzrsM7)6ok!;f#dt{k)Wasmey6c-VL)4!"
        "+S>nG*vs+Mo~@l(e`>Gvoa?Kfa|@V%Fi3)18JxP;SAaj3PIHo_1i?F)@sL7534#=b<VcuLHkdQagU;BI3bxhtU^YgQY`2nlwNuP7"
        "^_>ur>w;7wez?~OrkGUcfV;WvrZN6|>vN)hgPvJ8@vu1@USsF8BXx1L4eo#M9Dn$<wS1sXZ0R#aUu%1g_tW;S;8N<X=ny~S+5qN6"
        "B5#K{Wmfw4y*-`wJI-`IvM0{4zZ~JHwO8sOyX`h%tsUn(pSC=DY@4cRd)$?6oxygDze|MYbTBEaU+sZC8~)-<?Ec7J4(26i4-XG5"
        "4A(@9wH35+&Tu(jt%kU^{jooBmiA(vpRvR4h&INS1mq7MHl@pS3AJ5U$8Rd(8<=DJW#y9VPZqj<<WcVq>nj~UHy?yiu%RNh)OfUK"
        "?f&)E(tc+zF$~)o^jG#|JhN9_@R+MKbd6GVYbc+R3O9V!>@@ASgXVZ`bxWqT*zA&Nqf%eJPN(&zk;`DOlBU>Qy-{Ah^ZM00FRlLl"
        "+t;uD9GB5IFeD)l8IlJw-HstxNYKb*h`1rSm`oz1ZY;!@&5-tnlQD8wj#|tTD}?nj-FS2h&X$XSD90qahGh^{d#n!4pEM^V^dqQ>"
        "L<kmU*aEf%GYL>83<XhNQrKEgv8&eY_Swo-I|vpkCnI!6X_89tM1&kfH}aLt7fHN_#zmIW^_^EZQv~>DCFb8H|Cx`Xq%>SdoVd}P"
        "--CAqw~7WPJiYgfUKe_wIVm&{r<!`ET(XfA%>t?fb0uzB9(-~Ff=IOctuwK%s4A@ATML8>VMiI&f|T_Rj_)a|<er3@$z>#r9vehj"
        "i8I^->PPw<?&&FRx|FILhKOU0D;HxcjWGW3z&xy`9{l6S_`AtZ&FWe3T|8D#D4T4khRhmICLdPoWq&>XXajQ1(q2JrHebWk?yo*9"
        "?NQy7L_XfQBCOSi#bPqHN69QJzu?vacrt4|ci}0^@v~Jg><b{%y@TM2yWGdq31aTCEe_lvc36j|sB)(jsUP5ZLuU|)iwmYst&;0X"
        "$mg9s-Awg<Zo2<``{OhG2cB-yQxDX%+WmZ6*Mo23L46L&KfnE6Kc5T-aQp#k-_}dZHuKBIBf2tl@C`$*NV{->MO&zafvnZL85PAI"
        "v+u^2^<;YGy!T?aYyYj1#MjTESQSqfCTEggKZknBtXzim=bZO6@Jo<f3JkZBqk+qz<+!AUN4wfh8s<&?QizrCDh_n0W^Nqj7tJll"
        "kABVwys`@8^Lqy-=xK<a2HKM@sg$FzaVo-M`lraf@k#vuh=R?<p>lB{`Kyazj^v`?QzMBUBfF4G#oUaCQ|i>5*)1I`V}`Gs9}TtQ"
        "(%e64&c*fQ-G#HGR=?cpNbD}^0#WY1VJL{Cfkb%Nl7{N(-=B=3hKI=hcnNJ7&Zmosy&lhJFdFyY&*rQ3cv%0Nv2~#*E4+<I>ZZvq"
        "b`d5V$vDT`+2BViT7|)(&a`ei!}<2q(D<e9{p-oRSkqs|<<;U(=U*}})c;>V=*3XKmes34f32qi9$!xIS^ec1{n9P4)Mi_&Sf3Y3"
        "5M5O-hcBxGebk*7;NiBQ4oSSu=H!@s<t3sl7Qg)|^sjF<TlT2mTOi=gBT5ztRa-sqW_t5HGpmZcQ%d%x_a#KwbJ~ZSUZ=(*sk^Xp"
        "Nov7IePM~G6l<&T(#wVTj9nChT3H5~Mg{-@brDapqxFRM`-R&>eE-ib5qN{RD_lTOhcrFPXoCq=kusQw^F2Z5N|psB3hzVy5ITwL"
        "9%`nlm6M3ro97iok}5+1Df-E=lpqYzGQinFaDl~`0M8?3blSX)V~<TUHO>*%qBJ^OEBdXch5bJkxVQ;#-*ifSl)Hnxn}1k>=I17M"
        "AEKcZaJ+7g?D&h;#1-^sBUt}B#{o#Gydr^9oqfzzCV=sY!o0@x5(yOXOWVr(Ra0&O5%;nrfL|?iMN0egR4+r*6lKch7ke#NLFc=3"
        "uR<<{RWnBoczUjXRv$krBguNNmynC(|ALoKOUUOvy(A$o$ViO&e*sWS0|XQR000O8001EXH>CH@8WaEk*GB*VBme*ab8=%ZVrgb?"
        "V_|G(Wo>O?FJ*RNY;|FDX>V>XXLD^hYc6nkwLEKc+c=WnQ<eX~<9#?Pp%uw*Yoghj$~jxPojh)w*}5u6#iB?^;+Y~feE1QU|NC_}"
        "0HQ%sw&L7+Yd0d%Xf*oW4RBr8{gPyvh@HP*d_4V!vnj+TEW#?uGbhYqrx2+KOR-m0cd2m7C@+$1={u!_63(K?*AC7}3g_be*P-jW"
        "FJHWTk*qg)Q8{6;1h`T(FIHi>N|Sl>{J&+MH4g>A5xyu-QxPEzrK85)<#kqxLIG^T3eo7nFYt(~{&D`#4;LRke{!addoUi2qm%e("
        "bh3XqJ{9|?^Rv<DG(0{&IG)GibUz%8j!)0_BOv$U^PhjbI}hG}`2G9>rry7N;lQ6;6=9OO6UTGMCyRr_lQ=pWAD)~<N2B;qEas;N"
        "XGdr8d=4*;_s>qkd3btyERK#24~5tdLlGa0_76|T2ZzT;uJ4>2o*WE(jaifxAwD=c7{v$U{l&33I6NF39i4@dI6OT)7>lFv==eB{"
        ";6En^Comq)MRayHkD~dRIGi6u2WRp45D<?K#%GB5;pK}DpDw=s@#nj*k^`Pv>>^LEg;TABvj#SSfb)4O)=pW8jWfTae}~RFi25dh"
        ";eUreX94`rZ$Kwt949W4EKHrOUW+1$KrPA}Q4E1`17!Nk7qM74;uhEyRe{o%KmvJDzTTt)B{`TVdR+HoSVb#Btf2p+ge8iylB>(9"
        "q<t`b9Rk!kzYf!1oGfpM5b#1O8zNm3#wkD!OA!{)$}3#L<$c@x+vRBQ?A!30KYKj@eld#V+(J}!kx3N8Ws%n#Z#)>5RgrAG!4OnC"
        "1(|rR4|832;3NyRnn+9Ge9ALnm~aiGEfNt2WnLGN2#Wls^x~udUCxU;u$oPpRGJaub_4Gbcoq8QDMCeut#GBhq=_>*FAl2^#fwUB"
        "kPn<Ujz?h5YsVd;U8U{-=?~*zeg_)osr|wRlobwP<^7{$4>1mgD{&hqOVCVj;QZUvB2Q6F+zJ!e_CLe47UxBg7oNK;*QZx{$A~#t"
        "Cnd3H;sPyl&x+N+4<P*#sT-zw7?)m}WMTjcS2*~~Niy2skY?c<AR8&hpR<PkRAe6FvI!;5?M^CzKSHZi3tYwscp(5Z;Aoh9nTCJf"
        "RC}-$0(bHS_^EYWg<yBirk<yuzW;Uc`P1*eNQJp0d5F@P3*durz~)!rWQ}$!FIN{S3at=-)gUD#iPS+a4~%W8L<?ZJzJlK#IAoDk"
        "<@9S&3)H)$tb+WC9@Pw#2C>3-JX=eH0ZNFg<}j(m8id679s_mYFerpf8GU^UYG#30*~B5yW#h%a-7L(W6XhA8t%Z3DoCL}i?y9Ob"
        "<>bwqBD@(c!TZ(ox)cS%1BZs-Dc<B8k%h^dWf8^@1EW=*r+er|-`wk1^B@NUIT+9&s_>h8qUAyD)GSVJ)|TZE9&sfP{`H+IS&O``"
        "rpKd!6PBQe<pu(wXmLWh0X<5F?wtms(Fp!Kwi%1|LPqZV>S&OW15*A8NAQ@1#raK!<}KYpUIb<GU2tJ@N#7KpLdr!E0mj_c;T4Ej"
        "K+*x80)r;W8#siKS^a(mLSN;e&m(4kUuv+UFb+46GLSms<nk8`esN8Sg_Y<itpW?oK{isgAR?urWJb7<>p&2y&<MPd@0<>hiJp9q"
        "@d_Djh#NR><($DTUI_t%bq$-Gi{QH`a%7xHma9s0PTYnVyU>NY-?DERSW%v2p26+@H)<2?XyTwd0~#*fztk&thIaQY`;DTj0)TK?"
        "&mmU6eLH;hw_ksy5a-KzG77shCN*RvjN$IhCNC#MxpK}r#$=ni0^uz4G)9L%^Wne7x#WvQDJowh6BN`Tn=~@!9Me1Sb!ITdiNmF+"
        "JhQujZ(b_WZ5tr5l}T1sVFvNsAcaUzVnhJ_R)>H6o=ud<<+L5m0b&fyUj`aG3bspQAr=BWGics)TQ(*CvG--BB_*XJI}(*-kU=GJ"
        "(9G{4DT%=kqGc51E@|QfWi+{$BAkp4;|J#3q}+$$2Ao;!*@WENkEE}ft3eV&K5Ulrdti$^6#(R7=I8bS%wZR+by!>>6w%rX)~L+H"
        "Xx0|e3h65VlBL#tz!WlWDW`9Y^tCcl6jJk&l1A4ZbaA}4thaAmQ`<YYG3@r9AIu|L6%A91>3+Ky9BoSzqh}9xg2YYTZwC~;?;%Xg"
        "W8n>~=wckbdV^)1HY$a_g^)5F0<$vPEoh-`S&Cu<C2QqfqXj&U+^~ld)VN-6N;wbA9Wt=sO5EwZNfp3(QBGBDFu`<N?^AihezIif"
        "5i*bTR3^ic7zqBr)4tFqM|7qqpG;aWpZiiL=>*<GL_4QJ<q6(F$^!30ClfcS{#PrQq|xnh={{~@k=$ZR0K`~f+AP|et95rPtKotF"
        "jEPjB^O=%e^uN9pkQ8gW88AzEG2mz6TKKX~!5Htk%A?67&R1N3VB@)~BnCqX$ddFhJ`6bHm5d<w)z}9r;X!}V@{q0y=uG>763EfB"
        "67rcKETbfuz7N5L45Z1sJS0=tL}1WB(EF)dSBt$<w`2b6FiRG|a?|OcwE+|4uIM`F$s&)!bSa?rR<#cL+~9i5eT%DwBd#Gjl9SXa"
        "aSs$FQvh-gCNzO+uG9ueJAjwIL(kjFiwBN#TO@Ev@G!y^fCgf*&e5M!x)&Jb1*->e^UUXHr-%qIDJ9}=a>iLNwPMC+HycKoPqc$;"
        "w1G5%P~b1pwYIADD45L#NM@@%%2QmWrAWI5FBeo2s?wl>g7XTL27pPw+F+m#QJtc@+XhiV@l$RajD^%m3e1HN@m)JgxoZI#SA+mM"
        "=PATefKV!II0j)9)sPW&=TXKegH#DQO+ZZSp+j@-!8U?Rm&FJW>ePxE8cJD?5ud9^NHDHt)69A`6#5Mes?&XKxLrGKccbJ?6Evuv"
        "k`aUXz*mSmN^2-UQBRb2&`~bWt8HBY<^@<c#-k}c`=ek>+w0J+;{#<46}BOykxP5-ns<X*nKni)fWT7I62?Ol*P0oW-77~$wdrHN"
        "$|09-gxOAOt2-4T5K(IhM#e6+o1P8YSBWL4!ks(VLVeq7(&|m$8c4Rm$<@iNlDFHvy#+OfoUHY#tka5Wv2MiM+5g~vxW37Jwmp@z"
        "ENPNErCLoK<ZQoG{&IGyhi5yMF$iziPF=ia1;#7Yc(if0+mQAs?@4xuf6O4P=B15Obu!J%Szob5q&^;Ta6>bT%Y3@USrf-*s9)Hk"
        "fXdBd@B@+s4)B+|3ZTfpgO+HUYE|8hhE1oAttHei3A4La4%8I!ZOymA@`C1~-;mtUjz+4os#-8d3=2RPY_P|R&t;z0vSH98ZpK`e"
        "Htom=wUUlOXn;3G4zY`yAt4}B)YTWIZI<u%UP%bwVev<Y0X%`zl|v^bj}v)lrb+%Gl~PGe<F}Y#s0CthnSmeCCu>$qTc02eH>ev-"
        "yLdq2QClM!e}Ea3f^wGW7TX4jk2RJD7|4h$^UkJ<BLseNIHg4<gdsB+ZfwMkcL^-XEtElKV!+na`<mauR*~2T<Jh|tPWl{;?4-3A"
        "zs0NQz9*NQB}g13T6VOTI=jm98)}BdHF(S@gbd11%q@n{UYez5l5hz?=Qt=Wbwuo05wC28JWM)#-Zaac@5!cRN|<y;m8=i9F%A)J"
        "<(%4vK)Dd{<kQT2wzsCIDUT8v?+udv&K|5>)zl2u4MxU211eIEOJyN675(Aoe<&vlKepmuicF~5NuUTV?46SdbHWO9TML`9b`PxE"
        "qBXF+-PXr+%S>%L>eQcWt=aJY>C@{itns=uxw^Bef>0L%wyIWEg6)E+yb2r{rtL19GMyUjHC=kuScKe6&1ezaWp+m+@(pZ1q;#L6"
        "HMp#0wF=T?om6erkYYx=yX4AYO_@2br_R{SQ}B{??LlR%JGCU0@=lZ#;0@g|X&63HtZFqXd9&jg)h1N96(-;^uBkQ<8`g}UG>p-2"
        "tiC{lx(5yt%&o`~)WtqRRC)5r0#N*;J}vz+<*v>gq-AUh&}qn0Am=SNBBEZGDn^mHRH@^SSD1a=25KOv@+*;*)BVxmsddaohuQjG"
        "2_SYLwY&SJ>EJmuFZqtaA}j<IvG6uW&#Ug16m^#2Z4!3QhJIk|yq)$?fN#eK<I&0f4$)H^qDC>VW6*V0nr%%sb$)5q_?b*|_snId"
        "91^yJp2uth1*tmEg<X27mp^zkq{E5DdK4n}Dw2u{BDe4qpzeNCPN9!PX1)`~aey=?P?#%4qR`qsqy_|-8g=EqBjtKh4hpzP^BYdE"
        "Qc#xi=nk1W8U3@A*QCmN1L+8;spg*A%1~)&E95<i7yGp8Ft%Ze_DmVzrz}Ak9y-AEE{IlP1rZCP51xvewJz_Q<|p1%Ks8gh6{wAg"
        "ggkI}qa`<Ox5H2*%yG-9@@RvL*lG@1NK8__NNyWN(7Vuzp!V0R0<u*#u&iF80|S%4E=MyoN{x1&mvXk9Wp2|31mEP~q3&AUH6TwP"
        "VKDzF*`bRo)h2^=Q?+ICmx^4OF3N|-JOaGAqZ#zd*NwK@AQLy-gR+qx|H?)w<OFK>8s>0xb__7g4gXzHE-z!M8%B~u<&U#m9>s%u"
        "sj-eg4kJ=Xd_+WE$~a&`!ux7Nx{;-JDKAW@8%flYY))$$K5WC2WGZhN8r9RNWLsG6Ro`vG?imVTh^$9Nz<+lsn{X28TXVllYd{m="
        "0xSai>8HpzY#^zB+7j}>_+oMqRwno;+9uTYqeYExxpaQhkeObmbf0j`Z4?|x>qP!mGW%E_(4(Z?SFBlLerkI{?`BE2wLCX5PQU#|"
        "Zl?9JWn1=R4{WbJZ5!}=cLnUj{ph_7wL4-wpt4!s2ZDbu_r^0y6~?|Z9v%D6(WqU?NSp{e0bnzBmk6-gzOx;Z8F%aV&JmAkOH8C*"
        "(IICGiWO#)#@{M=0cB8uT_u}-7v4tI<mXsZb_Y$JF=2dU@A;`sk9&rM(E~qL^6QpGvZO1@DLNYm3@N(W<VnW;UszW;cmnW@H*~40"
        "e4}rvQ6EFP$0!nI*>Z)_LE$w4s2W1LcBgz6n0QI|F_gEGZ>zkBR{CAn2|~PHB+JI_LAGZEv~3QVNI3)>pd)YW2*~Mer$hS>@M7Jy"
        "2-=245apr@)Wx_m02dN8PT2%@6@b)kF{54SV<bp!i1|Zx_kiwlwhoIL-^~D<O9^zu!7=~bbp_M$pl<@2+X^xW`uv!S9iS0m>J~;k"
        "FwvbXn+rrjn&-hb3H^A8e0De@lG<_Or=w9X*4=}VmbUGJowrcOsVj6;&=HJwv(zSdRzb%ySG%U;y-(5rhBWxsMMSA38QWHW?2tu{"
        "qykN+k7d91FlpCow}RKXpvDnIU68rFSFW-mR0lkY*XO|GB(Qe4hIC)vKHTC;7?UJ}EJBUbRRfwIW}sJNlrv&rc5S7|x^&TV86Y~V"
        "f5M~nCK}sT?MRIL_%N#Da9D^ED|-<_YAM!vaTj2WYfksLpObfscD`z_!Y(CABl(K%hjWy4Gu*xHyEB0I!&i`Kq`V;I2Pb_sjm?i9"
        "&2MF1(v`M4r#<fC@#L&W*Lx;BX;`~vbzLLk`CQweFo|jo@!K~-w%ir5QAjDsdmuk>fosr<Ws%n9s(p5BRK;$cl8oqv16%gqZNRD1"
        "sDAr)8&isQ@G$TlT9RufL`RHY_9Zcs+P(Yy@yqY$U(f&3>XA}s(pr!WVvIyP;#{0;<1U~0WCrURKOy@d>VbpUs)tSDM-|ewv`&L>"
        "Q$^R(G!4B?{jhgK-JQGu#U<P6xv}K3(>rrlO<ktoIhZvEeN>pD^Ttm$=z1*7V!jM_tejW_=XG(Q%PX#*M4`cK4BAOX;JQ#*8jtLR"
        "3;ZI3JP&zqpK=4rYC-yroM@>Or2F_mOwT1ZqKJ+G3QtlO1S%yDf=*QeYh)$f9RdiLqyRnpeP`BkQ`>vTz#5C|1d1(GLTlgcJ7!jz"
        "+kZn$TG@{G!^~$fN!H}cboPNMpZ?Z%_?v3-e4!C97G+c<8`Ps|5X5-|%uy+CJ;xI|0vl3?uAa-M5@%IYm%hKuV*DmV{8cBQklaP6"
        "3>t92L_#2Ed#)OBZ9oFiEhx|3+mlz7_L{qCuugTi5mPKR9$m77Nt(j*=gXU;YC7dG(JBWams9zj59WW1uuiM#=rK%8Z};}->p#F5"
        "k_}I>^o1Z$0Wt7$CDP5*eNP_<ZsZ5p{L?^b7t&5ty{g;KT6lQLp@;SK-sM1@&FaeMX?AU$-&{g?CW>9<^;G?Q{y$&N|M>9n{L|Oq"
        "({CToFLYg{6Q3Ri=^@P{{rnSb+KPfjE~8Oxntu?diYj|77ybzXv&WD<HCtuT&z#+)k9#r%K7SXxYSWW!kl~xV0d4kl^0}jAwqwEG"
        "{Zm*Vy6!<rvZoT2Cy2pBz&LUG2eI+5y(w2vo}5nhwRg793JHvSRx3O1`{<c~+n;2xgq8JO?5|yVC;GhJ(I34?Ncwz|7ligzUi@R>"
        ")E>aUp0CcxJvYWJGvVIrlI?7unXJDvFeGk+{Lx%j|2^2R7zcE#C+(AL<J*`ByNy$g!8*p=(y!}nJ?zq%lkv%F%F!~`MXMd!bFB;R"
        "%$o7fzSQ^Z`@MQajq6`=y>~%33Yseiz2#=-QZ%nb`!DwGdOg)l3G{~}ox?Gjuj{_uZko0y{!AUMG4vEZGx~oLFr`mF2|v+@X;1%u"
        "NY{;ywu|#Cs6XU25+8oxwGzT>eGx&IQS`N*hk;7}VI-i!BM8Xgg8++xAkYRcOM{m${tr+~0|XQR000O8001EX`~+qRt^oi5`T_s|"
        "CIA2cb8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XXkl`0Wpi^baCvP~OKTf35WXAyABH_O;BM0E9tvAxlR!v9oVG_ZmS$~XJ<N=h"
        "K7M^iyLFt>=pYzL^Swq*(=4w(b$})p^uTmP$FO6xF7Cj^Ogx~W7S=>ylab5ixBl*H|9ty!-v-+@O*5a(X9K4IN6dTgw&21safU}N"
        "U+=bVNCy3mLcrFfm{}R+^mO@xzF>AKK2be*uSpCJL-O|2vm(re=|1HrLi=mUoVe1>XU3xltiLb6P&E4+<VYgjd<MEK4L}nceVC5w"
        "67s_Wj!f2>DnmBaOgDO1fW?eLdBgPLL~lEg%nRTWH_nd5gmeKPKmX^YJF{q5t5QY7butIyiV%rASoODeo9BnkW4~T6pzfy%FskD?"
        "Fe*ziV>4B&(pIY{<m*Y{r2<00=mru^Hx$V_Qj^$Lhh#EY(1!;ZVYfA&fPBzYVhS!|PTZbBdu?Ke-u+goi8Y@3A1s&@j>t|@RIsfl"
        "Z~`%9K#*dLW<3;t3Rz=l{|Vq+Fgl@k$M|M2W0(HWW&zFL(Du_nfqs%Hm0&c(B*kad!J#U~=|a=0%6>SVZ2N}$Vm|u=P)h>@6aWAK"
        "2mk;8Apqv;?GB_1003|$001li0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZB?ObZK^FUvPPJXm4&VaCx;F?{nL>^?UB}KcMV~"
        "lB>)(Tf5t-RXW#}Yu#eWXQ|zCZjJ|&ki|1avH)mX_42>(Jpd#@3RIGAQ!Ww-JiPbte&fNO=gp~19~imX$xV@yTb9S0E#>z_#05|D"
        "yAfeeWtPUNByo{2cw=!T(;^>{b^62-lCZqkra6@b7vaknFJGJxQQeljh?x+t9$5ZBQbF=U5=sQD<*D2eLG$FccnZmi2`LD#a#G~k"
        "jy!HyPB{Cmf?<{bI?wYEXSyv54j6=No(s`Da@IVyq?;khw&gm_T9~Rzlb0{nyx5Y2N*ZTW2qs8FBS~X9YNrfnN#!O>Z)>nCcty_C"
        "+-<rpVw&Bt?UsgdkxNcv*<ie8Wg$|K&u*F1Qfvz8T88x0TJ|LCl`+kWJOz+Qn3d6nip}71;4$7<TPe$I7pcKW77Y$msp2rI@^qcD"
        "BuditdIU;Yvxoz;xI}I4<%_>JV(@|H?<}9Jc*TO3FVq;(#<Qa0{+fv@ljD~!2z=69NSeniN|P}WlIu|zla;UrG-t>qFfk^#MUm-="
        "b(P0xW3cfO$ml8ZZ!#}(rXe_^qR164G9hRrnnF@k(lN?7w@2|`!q&>hSyU?1*K+dR8yFc(MuKC9P7O%O8QM>3m@}gi>^TxKPfKYJ"
        "|EXT2IncgEZ!=~Cf&Z}>yrH!Lv7vwd_OCV!&9dS#LI;f+2Q^ZgkaL;|Rx^@r8HgwXPb~5TB)iTEDgo#($9-~!sRzbIy&Izf0LLF#"
        "5RjucgGR9KQzPF`$g!DKN~Eb^<QJM%Y|44TeXm}ytw3J4pc)i1Aoqd>=YUW6^;<x`m<-G85OMYZFBC=BMICKZu|?ykxsZHkA<|)R"
        "GURRkVYl;Jg9(CK12HQl@mt<TWU(|4ZGa0`T3FMkG6fs~L@vS*$lfHA70>rH<XiSVBHq=>>hvNyKbgIqo_Ql<FZB<o-B)OXoAI~D"
        "$M0KBsF?0DypPjwA$r~h+$>iotJ!I^n7@4o%wV_~X>%iWu4*WeBo*i`N$9m0y9`>M7s6K3(hojcKC}7x^+`0nzFu7Kaa7cq6A2MR"
        "s5X=@$xlU==r}hbn>2xBsQ?9m1V~~~8_Q)1f+jfwM<ehC-*Usq7DSqXWXej08rfqiH%&4NR}5o3<-0RbevE+#EeULK;+ec~g!X*o"
        "DZ9w&mf71b3iNr*VLna;A!nk<9+)3kc3p6Sq<{%+;~~C@aaVVkDgc$QP`>Kr5MQy1Pl1z!r6&XmQvqu;r3`Q(g)`{U`>5+8qKJu@"
        "mQB>Kc?E7bNIBfzCn@*!i<oGgvZoY`rMRzi+}_$pC|+1}y8{*Kt*Lws*q~|xefeG`*GGTz0&A^61?LRou=nHad~tg6Hk$qdU+1&="
        "Y<V#~i#+nRDoH~8`o<5!4STZID~p2FM!>!o8U03P(*<{;fFUm|ceV_^B>!NnROYpYiY+5p=z+aeIi%BggPug_8ZD6Kgi#I!8Hhxp"
        "0cdl?O@=l;h-4D&Io#4T_Yrua!bX5K-y?^R2%%9FWKGiyipf`Cv{?OBAVvX^CCxH5W#R0`Rz1Qn<3R0)x)aQ|T-w^4iJHq@F6omp"
        "3s*oq<n;T7_Q_#jmV++RgfuE4b+Sj*v>~)h!g{R*8KFB0*zU7e4|yM2FcfnGBAD#^YmOy4s0)SlNV9t%h1Yi>@VHdeTk~<rNG+JD"
        "s?p_eUGBoFgsPA^<fX7fhzhAdNCF(GSM{<sZ^$1K4fQ^dn+9eAUefG)AOC*nIIxj(sF4E|XB9QD;414;aBQ`e@c((6NXWg#c2F1D"
        "KO1&0(Z_Iqb3nV|G#5Z?%XnX7SlK>+U}SU(>Dd=7gW?`paj6z(!#kEUp2m?{r34u`oIym$GY0zsk#uztVim6yav~w~-d56u7C()W"
        "a7jod7V5(ys#mz^h3J6B4Ca{GK82uK6N3A~?z*P#&<|?cAKt1W+QY9gx0dA+?z)z*Lr3^g*zVcxvET_h|INL!wEM=)nmIsqA5wmY"
        "b+Csa*?SK$=*Ep!2?N+Rt9@8E(TheziDiPbr-j<B^mT9_5bAsPunD(_CoaleltJ>sOg&&lHAl*Jhe7az!kyGY_tRw3=0-RM$M=y+"
        "G_aFcJ}J1j1x@@Q7#&6nG7ys+6zcnrK-My!+#G^DL{9}R_q0RGMwGb74imD81&K+NRjyMdK11z>Cm68UyGHiH?*x?~)Sj=hg{|HV"
        "<vAQR_I=)~`z<^1Yu`uL9*l;%pvwj0(mx)NXH0^sp!<S<2=>-^7_?m4gQ~W;<8&L0_t$8VVD1J64P2-P+`!-(1|G^2b&>@a(j%cN"
        "%de{(BhN8Wxm4D3k5nwH7!O_RD)Wa}QS7MFbh#4{>!;vG{%glxEmaz>_hF#GDN3w+?y}+*yw2apv-a>H<j2Lfgj&>pw#)!zJR)Cx"
        "uTLCDO;hhkZSu`mFo#RO3PwcJaUm~x3LgJl7qEM3vx{7P?9t^hHn4nSI8ak#Welb_scU1qe7sEabPIJT%O6r+<Xa&19mt(roWJ@h"
        "l^50RtCOo)=a_`aHQz8Kl#r`;tBb`PPOleM0jW-;$`PLNV0+==k@DSgHNCuEELQmT&)NKJh9#VLy0~0kO;0_22ELg7OU=JMi{8$D"
        "ygs>p7byf*{e|63Kn#9A*5OeHVYV~T5~Fb;LbIO`Gd^feg=Dsaf}&&`U-O#FE^2qhak03ZP8#p&1&g1T)9b72#W@_^2Vqv1SHpAD"
        "Lz9<RAgIA<E&lled05U?(~0-mbAnu)EH9SR>Dk0Pb_2{7)A{M*Y&QR?Hj|Ol&OnYs?Wr!>8Vb)%EEI5-i8ns?eoemqHH0t!hq3w%"
        "e(**^824+<2B`@iu9N8}A1xHl&puw%9GISiVKw}Csl0!z!jD;xXNk70#H*pye4vKd>tG(pdEU{iZwB4z_~5k3JY_~D2D|TcoT^{L"
        "t`;HN4<Q>$+2Dv1_&+ck(-Kt7qN0*zC3WM#4T*8^;Jj|%O<o^60D`_f>z3djYy;Mx|F+#V&<m_TG?o>`PDgE*x*uQn-S!-+2GeY*"
        ">m`$S9R^q<Z=1W}9WN@ti;MIuP@-H4?3DR?P}mRYcZ8hXr9bS$Ziuct-7p8t#B|zSQ)|z}0dQHgCQ>)5$8Dr|?#lrR10%|3{re$w"
        "ab*c=KXx=l*GI#~&3<6e{;zDwG;v&>7(XTxOgfIKVF&#*=RD+?{c^<xf({@i9%m49V&(#=UAuSx(SzJ?UJZKnCw8#9e!QApP8UC~"
        "8Xq<d{kn^;8rFuxj$;^ROsx6bF53Sfmf)SA$=#~_w&H^-&hd2J^i|Wm2Gh~Ts!G*;BQ{>IPL@j#BaXq$KoG>k3lA%E!nAj(nnd$L"
        "o)=Y~sQr6g3cxbP8&CI`1|WK(Y8kP`(V?umyI;6l@_oVY-<s8T0g=@QM~rZgqWK+5Z0&yjm5ggM{Yr7)n{cQva$lXfV6c`*Pb3;j"
        "HyL{9yuP1xlpY#q^lP&K)BAo6rwg%R$r~@YIr`@K_;}n8G##ZmU;^u^xZ1)AD*c^BxxSy!YlUxq$0SRIyy=Vey}BXA3%>SNG-J8{"
        "teSzAlskPxgY$#B%S3Q95FEU}Oe*|C4LCs9W?+#JwHZ)}7Ng5!8@rxuybrc)ro!HZJXXL;zw74wY_V<dsLS<_<H2;V%;FF(W3<oh"
        "_Pu5k-niY6r901}rC9+${a{1s-NKZ97(jrN5eLyWfaH}WMo?<|qN0=8`@|Yow%yC-9^?hV_=7g7KS;%icwJ>#6A!vVB5&##y3b`-"
        "H1(fn>}sx*D#Om-ZD_sODzlmH1Ubj3lXo2*m;i#$uHaMi(MA#Tw8u)Xr-An}8WLax#BY}P&kdgb>JQfP4>o^h9@x9le=TIpJ8;`@"
        "9Z}%0{~u6G0|XQR000O8001EXwiE8C*8u<k76bqQDgXcgb8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XaB^j2X=8M0Z*FrgaCv=F"
        "&2HN;489xiI}keASYix24IZ-7QeeZ-73;7ay26kpI%O+M1|=^Ay1x4?J8hgSEl(o({ZJBx5bL+^N^ng>ill+dU+ZO!R^7mt%Wvmj"
        "VM~nAIBsS502cos16gHyjj(Y>gXBPsRKXWQh%`wPYCFdPZoRW<(y(hmIb%=-a@H5JR<J{S4#=`LxI=X^+!(5#x)(zl<;Gr3jZ@NW"
        "ux(|boDCcupP=8dlf)N{nnssMOB0O^Y?%U9ev9Lcs_q%$v)Q?P9_8v->UuHGRe}!;f5)M6Sbl_aXR%BZ7=qo_!wRmhr_>|bA@vxt"
        "e5wqO2Ya{S^<eHjVPa4-YZPRCkXB(uI#QzS3_BW2@(4er>2b~MIBUeGzB8m`K$y`>csQ;5DDu5xG$^Fj*=g(y882MOu0yM{!-?Xb"
        "7rJjdf7FQa@P8?Wn3;(3g%?=7z-^sIQ(4(!<+zGJbttj#zpgMa)e&_ponSqlsG;>eV<o*JTAfMbkL_l!w>t*O8w|su#Ffh8<Ohwx"
        "p3#P)z0<_ml=#&T?r{NlPd-%cZcGnZ)4m7151IIFMRDuMX7Rsrcmrc8ZrQl8I7|sb<b$47PLtiNL?4>v`?wVU08mQ<1QY-O00;m8"
        "03iSr<!oF)0000(0000c0001Ua$_%IX=ZO@VQgn*ZEayMX>Mj^a%FB~WiMY}X>MtBUtcb8c~eqS@(j^UEKe*-RVXORPbp0<$;{7F"
        "$jnPiElSNxPE{yM%}vbAD^^GZ36`X$7bSufB<18MXX_~`De-av08mQ<1QY-O00;m803iS>kw?nl2LJ%%8vp<(0001Ua$_%IX=ZO@"
        "VQgn*ZEayMX>Mj^a%FB~WiMxCZe?;|bZKvHE^vA6S>2D@HWGhdVE+eJPg1G~cF`7B=M>yaS`-HahqTw1Z6MGRWfLorK9bt(_S*lw"
        "GyIS!S#~zXU4aA6d03kt^F6<rVUi^IezPU|D3u`x#aQ!16l<AcqGsg41XI*-DP9pFiDCUoD%z31|3qq99~qH~wDg`4T}#DuktE6O"
        "&FxLAWKSAuXx&k*nI;_XnP~|R`86X|eH<_Cc`Iw$9a!JfqL!lNhj97L<B2KWGhu!>vHHS!qiAhLe(%_ca`ons_NR{J{^vgwYj}-1"
        "t<gXc3nTBD;J;YK1Fcr7o|+2GA{(YE<i#FMPC7}O>H#EcIcq9^lc_b@x~6R_yC#tQmQgb(_EwsAj5beil#&W9aeMQ((NZap`Gtw{"
        "M>VkQ_QtM}@2sD`8=bSey&>?i%q!mL9WeuFe8&X{SQN#3x37&nRf<s!H{geIt3e}s8?M47x6v@Sae{c@kMn)}*Ra+{Oy!QI?p_aH"
        "yL?W3O(=iFlfQD}h5y^r$4amdu_S?w)T;zYNz!djwwteW^3^7b=oo2eR|OR#ig4x2%{O1e(Tu!&eaX}i=x?lQL}H#$(LmPJvOf)m"
        "L23*dlEf1y5Vl}#c6vqhzzyAZ41<Y4#=sRL7&!FAJ869SAbH*rZ_IPO$zREK+5<vDn#hQ=t&WLw+RDi1OIoil8pF}ab`w2Cxn|@i"
        "+6~NxYMM;&Q0<52z#u&yYp@q<lFZ$Efau>39aT?YR~s0GXXCTGuqP@FIWwx>bCISmqR`37i%5$MEU-cCK=kY9UYC&oS2Y>)7H?LK"
        "tiY&M%S~lvt|lr$F%;9P<Jl9g$y+I0D8x~Jjq9Fj)^WkAnhInH@xb^4tL7;gST^i|L#&lV0Pv)a`-v*rGsBclv#f9@MgiT=!aSYe"
        "R!I^#NR#I}RIz^z95B5C)B|**9ayKSJ}`bbno%UM_5+uLwq_bV3?R$}7l#NA3id#|)ZdbWfjoiZcOIA=Cc#1i0edjVs^dL3J{Qc)"
        "bsxAnx|&gN;mefep2nf-p$90sqXx_f`rSp2d7qO8EmOJ^ofL=MTdZ_{vNL*M3w=Kl1#fGR^q!@#HK(?hvRho&2JEOu8#zfr8F3T>"
        "hio;vFwJd*u@1b3-{uT1MdZRP3suhPXmGqO{w3+SHZgw84<d+F0GSX+;=s~5vMjw}C2aJWM*_D`c83kEb$3`K_=r33<B$jl0CG`d"
        "l*_4{ByOX^JgYi+;1KZn488p73Qe~GG<`|5vY?!~kATR{ce|Xt+`eBiRBOGBVy8aVbqgtFSn3%Uw1ae7Gctb86I&qIB7rJOaj)8n"
        "7S@NL1k^*%v7qXJc>8jj6+os3=I7U3Vi)vuVxmbck(E=Roudz-6AJeQJw{wGV2Wxl2hkt~Pb2`=zS_8qNPt$dPG=8<?$5VE?5zn5"
        ")&dj9$P-~(I>WszfEYA$;;dmvZA+3m(`32VDYvCLf}<>77VB*4vXpIHp%H@y!{DZG4m32v5Kp1Mor;)k&Y>FS!~=}&_sFjfKwQ@6"
        "w)D?SO^K}!bUss;^Oiwxi$X*08i)a?@h58P<16qUf_P7#px;qY-znG=%aLIn{1_UD5^Rc$;)wl0kg3P<eLxNvQeNt!Q`E2+Pc9T3"
        "4KH99n3Wfr^we9(;;LU8>$Qy?zj)TOV*fZ5d@GFKjRM23JVTN3<hml~%w6pGHGe$~`sRc-O@)M^ItK^ZIU%A4VlN}jH+_q~f1V!a"
        "2ojEeyywG@3sp4VoNn^YWr-9RBxDfQPNd;&OWL7pc?bQ_ffyJqxqy%z?4U*l9acke3*q#$NY2V&<l;~?7p2_d#T8M^k;25sAE98I"
        "Baq=*5A3;8V4LkDuCNz<SOL@Bf6GE3!Jl9W<_xi9h6;zE^gSCg4K+|9zIIPl4Wuf3Lg)Y^BCYTKohaF{WJi)gGX<qKv2qIoOU_Kc"
        "g;CHF_sLU!OF{Z*1JfZ2Pj}U3L>?y?PGA+dRosIiXGEs17Bn8yFKFS@&yD>nsb?u=6I@Jl84fG%+ui0p`D`-G9?D;s+MhjX!JU)r"
        "cs<S{zVgflA;&HcoT7={QDP3pqe8M<ys(7P+C}ni@tE?liF};dv1*pK-b9ZS6ZFMfS8CP7ZsoaR8rj34Yt#K-w%>daru*fK&~855"
        "jYR_dw~%|KyLMh%d(>H``1hMFM)?l|<^^ON77a)^0S<m(o;A*qp^i+<-LqQ}C*NWoMOR{oS(iRDq}-5;H>#fN9s?u9mK{tM!30LU"
        ")RBZ>iW6=YF)JYu?f3{N!|U4o4P?=)oG;DOI7B)&25awBV|dQcAr{^1ed#5poc~;$TZ_T+(#BkJSNuo$i}7J^^fjKq2!X$MFhXD}"
        "9Y1tMj(4>1+-{sTZ0bk&9M3BHo}bo{Z3Z^W%t^<J2SthEP|l%2v0aa#Jv5t^XSRYv4W|bowc&MTo3*I5J%3M(O5Pka*3sqa17f+|"
        "eDUQ)WGSBkDP_Qt-G8{Hb#E7ce(QOu<s4tTKdIyVgPmp+5O&6>Bq-H(^22I_8Mld_ljLXdPaub_ibixr@LV=)$@PBhj3D$n@MSh4"
        "{J+2XUkBGn%oUuCu+oS0${`EDYnN95IWhuv+!vvJ4zizy&v`^P`mdhbo8JLYO9KQH000080000X0CnPo8z}(*0OSGy03!eZ0CRF<"
        "FJftCZ)0I>XJu_|VJ~fOWMyn~FJE72ZfSI1UoLQYeUVLW!ypia_m+GI#5&3ix=6Cfs#2A9S5*kZ#HIoU4dbT1eX&zJ{ZY&YY2M>w"
        "-i(xT@6KqnKSBsvXfmB+Fx->&BB1R(<c%;r5k&TbV8{$sTpV{I50N`yXG%3@q`Z|<)>U1d34>^^+;-${3kiJQHsovz!i9vy;;0Mi"
        "K$!E63xOTR{bX1nm~yF>bm~+hd@r3M&Ai?7%W+hmW?_W`p#f`MZ1enDr{o89L9%KQRYLE~u)+yIEm)=J05k0bXDA~x@HiQkfGen|"
        "*?la-X?o|ay4DoCIH<|C@}96SzV_BZ!UINzPd*g|Xd|ZEj>2BE>BdzlrM*{5Jc!r266MG{AIr_Res^n&<6ic~urlh!^KY3y@PC5c"
        "l;8Qr`deLn0Z>Z=1QY-O00;m803iT4l9bqU2LJ#*7XSbu0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MSa%p5|WiD`etyo)c+cp$_"
        "50L*rWG_-HE0?u<@ZbTxWi7gOU7BG7ib9|z%I2(FM9E3|u;0G#kdjQ%a@KA$g4QArzjJsl-#PNh$;mHOBg=wkOo@uuyb;NaH;pK0"
        "mDM5@wJ2i2(z0fJnQ>ZH0V~ReHEY2Vk@7Zc7_Z~CY((6&wFr-oj*qTsQ`T6SeYm`44X;<CVXYF1<pSRmw$qZ<WzJ+F8_6^IS*u{4"
        "5&$S;b^7e;#p^fMuU=gLcJ(UE6JNu>>7{|UVhe!rV(p!UXHWfrsTe!ON*fs0rBb6CDVemm?(&|IcxX(OMb$RDO{rL0#A{x_(C>kR"
        "=eQJUDq{&8+9n5ZR*5>s0$&MMrmVtwQEx?Ze3Xc|OhnC=V$HV_zQZ5<sMl6CPY6bZ>`Fk{k0Q@GENBFFw2>XY%&Sb~kQ52bJUKa`"
        "n`5!88<z8CeSD-n*W6ch+is(erzP8zh|c&aEDFY<rPx~zQ{22u2}oQL&hb&4ai!SR&mZ2Eb?!mx=Q3$C;m?nc7(TEKdoa4rcOgWu"
        "6LPb5-?Em;N_^k>Z7XV-N^G0EXZ#+1sYR9XSYJ5oE=$-*F&0IX9#Mp>$kG6bmCwmx0TY#)votGtgCsZ?v+sI~=%ZT1$MeJCv~TSJ"
        "D8y+T!@th@t&poVA>yL%Q53mo0AxH=7D9P2!1rwfQ69IvPCEPU=AApQMWM>tXR{wh3zm&rUQ12|wqSQ*Ro;3RewYhhc(e0>ZG@<x"
        "YH?k+!m_;<D6@jyu}2+ht#0bZ8=xK!xAP%l-N=sRk5uX}s5+kY#qnInXfnE5nrZRbAkWMJltxjY&?yO@OH=eFe_nMN&kLO8t-{hv"
        "Qy5|Iyl3@FS?=iSCejgpEt5nP6m58o^r6OdO&h4b#yk<O!k|l~C+yXe=fn{`Vpnw!BUW`;m6Wc1jk=f@Bg~@BqqX1(a_?Q6FCh<{"
        "9mnV`OPrpK{+EMdWTsvZ*_)KTLsA7jmnsC+_5(XT-6^zC6SBE&=4=;_Zi%k5<V%^!=H5EcF>JI#ETb*Y=B#5;M&!w8nWa2NN`(lM"
        ";dN9lu_W^>f@Oc{wX?G)ey~H=#bXgKho_Srqf!d+EGutme_%i=;u4%td=_=taRgcar%56Ey7ajwCP_+R=yBDeG(nEg#Zq|(Z3pOh"
        "#KJrF-C={NqMJ=b^fhO{uLbO{DGI!*viqL(7##eozQ<btqTs1NH8WV7v~d#-r83D|V}c0|Bs@{tfy^CmSqX>@qQ38fseJ33!AD4a"
        "f|yRUi4^J?6mbn6N*GJ-X>h=Q%d*TMLY&o!X#OMBU&GCI1uSegxg1{Da@Lb0H@J*wz5njW^p1nb@KvVPcawN@6(%wz4wk8dE5VPw"
        "g^n?g#+sdjR8Z6!&cicys(XH@seW|E^_&$A9JQYM%%smZ?ES9H9DtjMO3<{wpJX9;?O7KMoI3?<DY;r)ayY~_WwVcDMd)jgw*MOv"
        "5`G`cJwCec2+MZiuWx7Xv|opcN%ul9cPP?0xC>4{P#ub=<;X8QL23D*O5FKPLj9t-_a4;b2s&Oww=1H)-)P~N=E3V09aqr+!K!|i"
        "n(xMa(6egeHF^e^MK>9u8y}c1{n&7(9o_kwi=Lv2vcmJuG5KKWl;PV9-~2I^MXNztS5j84a0Zq2cu$o9!Pabl%gra<Hz!BMMh|=5"
        "N2B}`vI?WRSeQ*srQcpmkCFe_E>r`pYCL96w2m(n%emTE&BB&;y2eAJEPCIiu^)lPgh{WA?(3In@u8g?&ICAhOQVW<x?1x}FmK75"
        "c&)26%0w|LSKpZ#Tvr^zf}Fv!Xyp*r*k8KiVPvQ20#gstWh)qo>?=9ff1N6)`sf1$#~z>EVi*y+ot~Z}8xtARMeyQ~$poR;2t6k4"
        "u&(smf$LH;z_(SR|L;fV-YH+_Rz2kO4@*lvvO2Ya4mI*lI<g!cS^at}y_3J`JErML-Wc8T;HDQeEyo)JTY}qOaC<<|rkN{YJcC)V"
        "ch1ndsps67n@sR98GaLPrMUHM4~@HjFmC>S*kGVl^d{mVO)i}4Z3ei(h5r_goUr}xaJH+S=~?+-bVMt}wCmkCbzwzT?{8ei1G%Ue"
        "L%B7~=pJPjfnit+%%8WJcY~K}>pt|dJp&i_<CvbiY&E!sUHtYW=v>E<4B<HsZEC?T=KJ;j4f`-fLYig2lru(lEk=(Y=Rx;wY5tvm"
        "`UhL7Kas7*SHx=*MvvWUJ%t1!k|>o~=KW<O2j<PeJDJHMJ@K7eOw23~>OsS{!R<I{Df43o9l8uvnKn7U^Q`0g_oKz^+&_41Z^H3x"
        "0Vn@{*Q<Jr!Q0OSO$^09K@%F+j8LQewDkpSdw`afJl@#cG-w(o*=be%SGS_WxI`tb+Ok!6HJc`{Dd_Hv9icb)qdD8_wM|Tf)(-^z"
        "k4R-PXP<%b2cm3^gO6DMTUFux5wO>mc0HNtpJxXz>Vv~{2IeZA&S6r0|2MU<6il$dAC2#MeDpU^O9KQH000080000X010WM3cmpW"
        "07C-+03ZMW0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJob2Xk{*Nd5u#`Z`&{oz6Z#EAbe_p#@T5HABGJ>3k*Yd$bwxBMWSONvM7*D"
        "I`7wyj+><`I$*j879Zb(bX8T4@7_01n8ML(qzvBq9_0=~6ph&{bW8(ik3cCvOz9}fPU8tdFm<c1KL9Ora7tWPRkd6!7w3}T)cNE@"
        "P#_M4Sg%i0(({{@`TkGQE_UQy+T$>|TD<GJGwPiBfKHE$-$*y;!g$Y2JXJ0iA76AnUvQs21>{E@h{tBR06U(t;!kb6nD@wqW7Y;t"
        "%^bzKqBZ&cP{@KU<0*=n@**qX|77(s1kA0dQ<P@bSZ^`Qh44&rgM}nobo@NI-drF$*rT?@?ab+BGWzbH^lxGH;Z42Mv57n!C)Xh6"
        "4%r&oAafcUIN)e4PslbNkt41pyCOa5TIql}{z6_qVchpB>rdaeP$)$|))V_n^9`<MhUfjjtkH0uX@ct%88AeZ8n*0WHs14kf%8nJ"
        "pNcKc1axgFibcTXU@URfQ!(Brm!b`L+D>Nxnbfe{E@6P(XwXs4pzOTXe<{Cy7`)({3v@I60o5H;Uxt!!FcGg1jmqTzdZujaPEXlP"
        ">tm^_+vVanP)h>@6aWAK2mk;8Appoma~4$u005;6001rk0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!3PWn*h_Z)t9HUvzR|ZgXa3"
        "axQRrjaJ`}<1`R{uhjoxFfR$$4e-EAM0tS&Apu(HwD;0oMQ$=_O%pq^-R><*|955_r%l=(dwz*B9?#EjK3fR!!CO^&@c7B!8Qr{q"
        "lQnwO&R7TDO6_X2P#X(=i_lplnbKH+wB=TLEdAcH+^jL1&*rm_vQd@v=s-3N?C`~X>$b9^|22{o1gT%3G~G)Ee;Lr|pN-u?Ny-kC"
        "CMU{4YpQ+2=7bPzQ?;G39xTphwKXkt(r+8JhA6+M-;tU(6j{_9Clj|>RG1DmADPc8tWRkcT!o_CV!7*#(mt8bfYw@;yS34{2#)jM"
        "Zvo#d!0l95Q2dq4VQowUOZZhbjukD|joNH|!8M)k&Sxpy{R9=2J?|Z`w)#Z-f-=V@Yoy-n<pzuXSM*w5+6{Wr8sf>Er$QHOC`o%A"
        "{M1oLs5?t|?IpaZ3ppHQv&XZHJ4*6JFAt>;_A}~f>jrzgR1MmANY2?(4=X+lO9~m=HCBZ;J1TSbjGVNHr;q*TS^4d`?1}R>;X<Yh"
        "sI6>~0k9OE7YnHHph{edvfGOkq@$M03T=K4MjoVU<hnu1(4kLyg^Y2Qq@$o)<($$R(ptspz6%k@v^n2Xtti2X4rfdICs$9CT13mE"
        "n0V1pIE0T}%lFnAo7CdZ7{VD(5TTgEYqYY;e|S2I^#9s2ggG?k!f_ZbN;4LMI#{@dikOU3N_#o0jyoGO^82pAB<!Z})iVBv0J2_s"
        ";qzv~*D4_&@gW3A`L@!R+mPTZg>=lpnY}F$UW1S0VgUH-wg02IwtkJ!>Cgp`64^vU$`vho8T?<@Hb@vjZ?#qYu9a8BasMEp)p@0*"
        "S4M*))}e_VNN(KoNG{p9&1$1bt>Tv0Fs-W*$av7Yeuhs?yg6n^&U`eMNbC}(;sQ{8p^J$76Rz_sBybha%FHj;J09PNrWCHM#b9qB"
        "q%@k!Gv*v#fi^}qRt-_YU_F1j3h824zt~TYHlB;D-ZxG7y@63H8u^0OJ!Z*c^)7wP95!_!z1Jk!6?Cut{qLdw><a2xOMm<!&Dewm"
        "hd4ESF=D+3iiOZkmThvrP@;D-GGVWn`;B-*9@k>VI$u~}d-eTO3O@oDwl|;NKG1KjCg+~c1>lj{9U?@FpnQ?$jb==Nok_xmt`er8"
        "sV+nHOIW2A54x3M9QbmM44U^8BHrSF3lHHU3eST~n_^>Sbw#JVP8>WQ@VA+GSdrEyl-!3s<>cF^zlL9wB{%T%{li^LsfV}1C=h--"
        "(F_~K_b^R6!Jk`6)Cs@cKfwJT?_-g&+YUyyGPYGDeZVmudf@G?9={$$cop@bi#%SHi^+bt)e5qvup*W3hAVZ%#(|p{l0y~KTP6<i"
        "^~hE}t30(FLplq_s;EnPLfM`li^x@U;!K)j?J0N}{4p**L2pN_lBvohEKNMbl^Sx1s(g0eCJfHcaO&tya^mK}GCiUG15ir?1QY-O"
        "00;m803iVRC&n9a0{{Sn2><{u0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MTZ)9a(XL4b1XkT!0Z)#;@bZ>GlaCwzgZEw>s5dJ=C"
        "|KU_5l!!ERFeX$~Bp5J|U?A8xgd#V&wuZzGwgc<G=ls&d2?a#HB)0FKdwh4#Wj32V`*Z_$;ZjxzpN#5C(<y`xN`A%Ca4F!OthyRu"
        "BNe=rVmrT-!r-m(9+$Fd74A@IGR;p<PEUfT2Hmx7&9MR-Kv-k5ZxP<S(7;3mdM^ySg$i4<11^lDUBG6IRmH^?bjwO4z#D})*W4IX"
        "Kv5NQq%!M=qC!#b0jn+Q%z<8E(=dyMTg4oEMwb!N0B_nF8>EwpP-nB5<+@R_v8-%BGLXu^`_{U#+8&23BUQN@uW~`i8icTBrzd62"
        "w6^!WQmoyD3`ms-k^AhW%hMB}hvNPtKmU~+5Bf}Pb-=fe=$gX96%K9**B$vOkQ=DDHe8fu8Zq&Ch%)^{mTcYCOzqQrKyKJ=;V3QP"
        "9lPaC*Lb2-lG3>c!bI0`ja=QjsCcuCTZ`7dbDn@kn_4(~^zg#9+RF#efnE<DT`c_i{CQm02)a_Tm7Q6_3;zMFl54i+H8=abPx~QR"
        "6>gv?xZtKJk^w|xy~+NZLy8P5OC~#-4S#%bo*lh#a9*50&W^C&)RNirMZO5fDa@b3wG<c?m6`=rlIA`2bj&>;GbX(8#>BekPKN#W"
        "#HPOi+4SFsG}`A5Trw(%RJ>&MMCE<2PH7$(70$x)6#2`8aWW>I4RO<P;MW`#BOSWP{vv}q5?#s`?+diuzJ(fpU_DKj^Eu9;&&Qv`"
        "zvTj%vW<4=<We?ks*Kg9m#$N><pCRHcBD>AV4ZzvkCyboqPQ8XIn%3GtkyWaLe2K5C|jxtFm`=S%MntmS7$iC`0f5dE6y_rCkdNr"
        "tQFGUPo}EBNF|bqpN%)9X{q8~x7noQ!)*N=+kGJpw%Oz6kIy3!kc{pBrZd#wI{d0a=k?XsZdxd8C5<wv_EZ_$lkyvZ<#81!Q*(M1"
        "AKXsO@vn<E^p4RYSu;~ok7)c})F|RPdF;+$2EfpmzHk2pI>3HJy04g$4&#*iI7-PNh9T_H2yu_+qf8HrLjzNxj%hd<(l(L_--ILp"
        "*X|cQ8EN*RA7|6r@ZNA0D(|)vznQ@zn+~lxoTX!+h-3eG?>DI@<9H*wZJ)>0AkgFhmj1<X5Mt7wMfS>-Z6v#q-7io}0|XQR000O8"
        "001EX>gOZ-$N~TWB?kZiB>(^bb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVWo=?*WMpY>XLBxad6iUAZ__Xkey`O3aO8y+wb{gD(GX}<"
        "0%=32>v%z^a_#G4aO`k)R<d8uIZ4x{Yd6uCRKD|f-}l}5k|eo4e>($7&iko!dEqSD09!{2LZ$89JNR}r2OmV`!DAsv1We%u5tNhu"
        "?Bn9{^OxoPa(TO$r@2a#BpD8dgAKVHln6o^;XQiLj7(vF6qDMs;7DevwIDoL8w10^hc0o%P!DJ?mQ><+IH=EHp080UX7|k*Evp6k"
        "(uCP?0Q?IY)e5b23TdUV{?1_SoN3}-+&@Z)f^rN<eg@ixHYF3pIo_2damM1QVWn^bq9{zYivFx{$6{5~G)B$s7J9~F(vIk)iFJHV"
        "$5tj^WhhjgbEPc`Y7^c}I)%pta(oQa_s`nH4O(i1PB*yqFO<Q<8aIv*q_eEIl!329$Pio+bgmze(r7`1_|(({?dt?#W#%Y_WyS}}"
        "akH%pkz_qNj$-c(=5efKhVpme*k}+&3*X=odY%Chw6iH(3S$^MFMWWGGsbOU{<lEV5#Xjk`I_gj%9%T9Id#@l(72y=`kFs_bLeQg"
        "MSYirl-pY%yt>%Md*`i9H7}t(!gpaxoD(@3B|GI;*AM|4mKhHe{J3{gtV<)Pn(T04t5wK;LXDdy<CDDBY(;c4`u3F{;IkYSxIyCf"
        "W1qzuA;L^IguPnFhGh@W32pFy%{gI^CLX3o>w}y7r{qHq{1pcFHu2mj$h1;u8>2ebEwNF**a1(4_gWq!+64M{c?b`$hw7-z`h0(R"
        "{G#HUdRIWV3w+a#lGEbj@chc4<r)Ro_&ufdK7xxwg0{@jjK)tpv>TK@aX`Zr&-btgmkx!ttBNnpo>Dn~W<X;3KGcgA-P*X5+h@uH"
        "AyLOVf;YY)9^k57xVuVcyZ6&Asm`$DFZzT!!cdYuz70o5-o>ObU;@dmj1IBzKTt~p1QY-O00;m803iUh1b)gA0{{R52mk;n0001U"
        "a$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MZZfSIBVQgu7VRUJ4ZZ2?nrBzFB+b|5i2grZmd`f}T>0!HM=rCYNhoWndbw#g+BGHL9Su!Z4"
        "U55SmQGO>&x&rgbAgPb9k7SZ0w-;~cAU7y~lup|KCAlZG!b${T73ABnB?K#ienNOgz1{=}B&^So(j-a7qwy#wSAY^hNF%&Q54tEF"
        "g<cem8d-28H>tHCJXqVkUg_LPVb)j_B9+eOdflQdewIkOKpVc5D395DDe~MI)xNkw5u?6y;U0zBEr=XNPvg<YE~E*={6c&AKwCT;"
        "k182lTOG78`WL6`?9vE3ACG`P>|&UM57Y>)bP6e3Ikn5Y+mgge3%}vcS=={zcBK42txAI@Z4}lv1AVRV8$qZl4;eRZ<LCtk<i4Yf"
        "*$n_h2%H(qvLI-UZI1f|`>RH%J~^(p<)^+W6Z7F}k1&N(Tf>AX&DO^v8ZGsZV^iQ>`)z>=Ry)XrWP0r2_TpVjtwHQ_=1Y280UoQ%"
        "Kz%yp;|P5`^CJ|f3!yM)PhOH<VW#(tCZ2gYKj7&ZoPT(MPj=W#xP%CiHqw-<h9=X@=X_5AXgr=|M2pf$JCvr!epeJ2NXr9npm`8o"
        "s9>v+-wVZapW)2`aun96gxqa=B+p5$ujm<{T2bJ%m+j|2P#mPx79eQM+BdC9HBog*PQty?&wmFHi#GsCxS*LRBi+TyVA5+2JIclH"
        "`&mUC;U<a2=ggoW%Y)q$IEn8Eove%#t>W~CU2--(sCqxxkqnLYtS5BGV`L``DRlHFyRm)+vtQZ*^M2$_+}1N!hI4J(Gh&=j+ODVV"
        "viLf?nLSn%WMDoI1y_*uX}brOgdT|vKgX@l-1Ay(8Pz%}X7_Hnm@U67W;c)LaUt@<xLY_U8GH&4&B<!}3)c_b>^H^bvt@&WutCVR"
        "6$T3AWyrzs!P0?k;$Bml_|NR#^UOd%!IzVwpC=^G7-<c9&rT?&VSpQHc7sAIC<XB_1b$GDTeZC*l4-Dr(yo-vpb5>bK6cn9@q{N+"
        "Nbc$?4sLmSr(q-N8DAk<;81n0QKbnp;SW$t0|XQR000O8001EX#-x%R#|8iZixvO?D*ylhb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aV"
        "ZDn$2Wn^DrbaZ8IbZKvHE^v9xSnY4)G#LLrY5&7R5h#H)+fIPcphy!8IPNyxZDU`isd5u{HBIc)cDm~c@wfBXP8vH+uXhq3s5GsK"
        "pV!~>>W)UEtJAY{;B`qQ%SZwE|B2>fqBZ532@aCZsiZuk32>o-z;CoH2^3^QB_37<$!LkR>G9F=(I0|w4NA;)N!|hBIj9AZG>2Cw"
        "_fSdk7tM@N1^5zJSrwL*=4m7KM!dO9;LYpH<oJja&DIoN-H(k7#-C0yk}asRV)0A1tgJRJWJwD4o>+5BQdLNcmX(BYxH|pj<m&WW"
        "{QC|KyrmR!k*Sl5o1cGwd3X8q-Jds?X_>=ZNYD!k^c~5xF*L8Op`Z!Ep{>e-l}uYD3Pp`ZBjYgoAfyH@WVQ&OQqF-W;M|ZMALVoo"
        "vzit8^e<!ebj=i-v4ZJMbbJK(L5M|8<#dg{<mZiyJIa-iiKEErAqqu{`Sh9bTs)tH!RBwc7M#Mw{zIY|#y<s&=Z+D*c||uaEDw#I"
        "q&4aBhgLOYq*kOjaVluOC4r|x%t%pe!1`6t)Z$}Avw7%R%5zr2XA^h<2zuaCchs0=K9Vp+;hq#Vy_8bOXmoFA5jc#sS}G`OrC~uZ"
        "RUT$UXNv@3Sk)$DA5HdEXiL1u6np$|`e<~9mb51yG~}{z-egCMkTn4mdyk2lvpLp|uigs~Fl7tM2FO;5clP51PCuyn7KyzgZP+Xn"
        "GYXYZ%oNX7bwT-)Ucj)IXOE7^ahd9zG&w>f;ZHPj+zB$TSw<(J4EM6p(`7Zv*vfmb*$3y3#;@+}d19S^JcN!1geU*YK?Oez$cM2n"
        "*{hT$SJBi??t^3bvWL`6h$4zxkw%r1lS!eJCn(IXU9ObAt1#Hn_$+}I*6~?`o}{{#9HP!&7lw@JDDo{%LcKADhLsbP)5Fw!4tf%B"
        "#I9Rq1yKra+SY$%hr&ghZV~6{6}Cv!-1U!JY{Er1EnrrNY>A19Y4_XP3)ev@ksOBsQJb=)8V3$6Se{evjfS*otghkBVLD};>FLx1"
        "#UPP%k1gZ~fo*!;QS14@5l+16f|6XhMmd4+dxy()cz))cn}e;;9`C^(Y+?3tTGbVmQJii)jk_}}ijwijb20IqN;)@^!02RaV%+Z~"
        "!~J9%Ta!sR8EhNa+fEwOw#wm)5VAws-quuZz_GSUa@JPSY=cAJody~WvZ)7!S8n*!qw`Sm+TJc*=fkA)XMm>lts!e4dz3BeNY#Gr"
        "QTRP>vfGz-{Ol?N$)gZhEW?+%2^%gEad_m7I&01z2+W5T7#$S#nsKbp59Hd65jFXhAi1ZkAD@-A*r2qPIGnKX)fS>XqOH8;N66d8"
        "=Ho=J|9g?5*1JN7T|EqGgQMANoxGc?+iQG?Iw7=DlQTO0ZWpaj<oX=`#HIsCYj@zaDN^hnwrz6(8f%vmQ<MrX2e`81M!dk$ylwMo"
        "KfNs(cFXkC@8T%?naF&96K_#bxM5Q@xZK)Y<)1{he^d5#?x)(fE}KuYYod@T4)O%1UYl{|x%DdOn7slH-3iuFD?X(5*5!a<d}cZN"
        "ZEuHpP@B&@*iGYu-Tc<bbE`J)yu$8nrCo!`z*OlE5nbagFdEp)V`Eo%8d!e*;xRUEIg-%Gfi>6js1rPhrxAS6>7h%AbK2Y^ai&)V"
        "!&S!$ZZ^Fg*PGli-TCl@{-wEKw3i#FG80^zdoIytyTTV{)zD%|d7#XY(RUgq`-V#oVviNKy5{BfT449l5W1vQ(d?pUo`luz<nl&E"
        "Pf%};rE5l?qkgT0lRF7#Z(joTFs^Wgp%L!fFNR0gl5SiiJa>DyBTKf&6E~!yj`u@eG`f=A-QquRWWC42_5a4gRhxuv!bPh%#Mhf+"
        "(X#K^I-M5+voFSx>T*%4mDJt)RIO?vDSjy_;r5!fq_i?Ac_(Y^DPMz~fN`n?a^-qc`J?wqin>y*72N<oIoZV`G<%s!T2Z2LiPuzO"
        "Ev{+gr0o`jab^7p^cXesYYTGi`2{KHo=MpKlAXQNUm7wYy~4%@R@OzdYSBdSw!n6=HK2}P#3lya%pZ{HzNDr>(tJ8+MG^hexRQVq"
        "I~p@SAH~5!a<LXHhih`(ZVhm8{R_^+Y-u0nc9Ll(Y2lhL0#gyZMdq-w*9s|~Rl3*3=LIe<QDa6*_+~dpl7=T~Sl&~-Y)=ceq>(%D"
        "?ZvX15gP{PJKg*Qio)YP0G#egK|3>Z8|@${fuaP~!z{hpZ%_7Drmb7kyV+AMYSq6-d`$ms^I)JWcwd_zH-dFJYVAY1e*sWS0|XQR"
        "000O8001EX9+RSUu>t@9g9ZQqAOHXWb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVZEs{{Y%XwlomJaz(?Ae?kCcBH`6-c%sSp<_2p%e0"
        "iqsTD5j-Gdt-WJs#Cq4=UAHyj<C*m(b<(r~EXDC<E@#e}8AnmHK6<f&+8Wv9o{WY<<lfj8oH9OvliONRP9C)IbdBIiw?I{`utM!c"
        "=RpqEN3Wjt7D?SEQ54N*^BJ2N>w%-?o87xaYY+~!4rYZl6=YPHT&NPON+h|_1(h&(Fa)#|(p}b=KVa`I!*-*~+DKF%8(%j*hD*G9"
        "#=m3X1C0CLE*C{%lpLK;QTWE<h4G(J$o9lqV*}3lELXxgIPRU_2j?!)LhFPt8imW%d<J|-EFeuulTXuyLsi7!82f6kmeD*MuHiR}"
        "WwLc!PTp~ZBm{srU7UbrOoKXS3~8VMZ~c&rG*_OIbgO~yFQ}cdF$nKDH;a~5!u{NRyM!licOFd8v7)C<U}fxsGtLX&kK=RxkKgg>"
        "$7|3A%D|%+wnPucf^4;@C?7yt$VI-vzCFgsR*=JkoG(IaMlnR+i1Yf*Hg`snpLE?S)EGO7mQz|JcC;u{l6Q+;D&xD`hlskf=mXgU"
        "d`*l745{OZz`a=T8!n8-1J?IL?%MvHAdT)=l~5tmE=RzfID8#{Wo1a`wjWs4YXBK4qf5u(d3Im^pCp4EL*8v`#qjC;1ca7wzJ?9^"
        "?nzS^dn2qIS~Uw}x!M`huI|YQA1ZtGXfuQWLW<fmuCs8n#|=N(V*AI+PaCD6oBeGxC0ah;kR@G)W#~Myjm4{mk&V^QXQB42Atlq="
        "BJV5GDATH2AP1L2KzVQ13|!pFKdkG78wbK}n*?X683%XOhNMId(%fiegao`L$IEhqb~22Q)IQ;Vee^uA@L~J{bUI%%?4lhN3HytL"
        "qF?>jAww3NyCk6JrSGuH6n`Ry7N5`l08mQ<1QY-O00;m803iTvMhxiy4*&q9GXMY~0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!Me"
        "VRUJBWiD`er5oFB+&1<-K>mT9eUSoLwcWH$U8vY?H(e~6v_TW}#V`bo#F<g8TSUqBwCH~Oo^wb_Bt?zgb^vEaB6)c3H!_>e=ErO("
        "?>PGpuj`DxWJB2;S=kJH-@`!LT(LfDineC=yxboK5iy?adC#(8;0;cQSktnB*InCZ70a{yz~kr7o<I9=;B4%~(DST@@rQwlA<J)B"
        "d*2B5?#=6n^<0$2m{kDt?VEQDwqZZF?_S63r=~ixg6C~vZuI<FDPT=kWjQZm_CBlm*{<#Hv%X;BkoCM^Kl}oy=hawn!M5#iV1idW"
        "0-X1)fOkL(;3ojV&1QpGmWcPjeiaCR*8`pMO;v8=nh#l#4Viky?szXs&pe=`9s3;iR1<e)#f6$S9J{jF>ybPYgR$+<_W8knikpUI"
        "f;D=Z0Z<k|W(}KbC?u<wAAqOdwau>FN5-4Cz_iM<Hm-3%#Cw`4p8&B`K5NLq^Jfe`Gq`_EVs=OVE#>njJMvx)W?feu)4m<D{+M>H"
        "NO`?gLm)Sv4e9M2Jcq~uLU3E|+B~ZWhd3u%G$yYfI^LHxZ2rFEIcNwX{J#8+uPg;x^I_r;&HiZ|x^alud;akf{=UtI{1CAZ?JaKr"
        "PVqDEva)~ukkwt~g^IEV&{dUzWHe;C>O1w;w@fh_n7-#lI;pP)upP^)NPlg?#?re|l-sf@hocV=kW@AxqqyDv%5${RiM|eLNlB?w"
        "0N=OW5t;h@S;2R#A&X8WKLY`@Ud`-_*o%ma{0c0jtysc7^dr3G4T*Dl`#snt53i*3z*xQ?tBMsU5VQfoz}^_B7DFQeqZ1us2voLU"
        "yI->%2&3(fEUWIbqhO-|)N#|8%nm`8J8e(F#?;eI$UaHf=JWXCtN47Q_9qR>O2OIBBQUe(ulv631MRyEcg4U*++|e>vIBpR?SX@0"
        "unZwdB<y$-Dy<cQ4?r<lZLqOKKx2SYjPe4eAA;RRT1@&oGB&Q-d)^1(A2E6m4G|C+f#8JP3QWlg5fCJB6<Lv@GYp8|Alzg#4!g5g"
        "n=riI;D8d_=4N6O32WxVeOZ8GOW0?ZIEYBF1Q_=E^4kpqz2ofVt9i1-Bpy5DM=-yqF(7lBB7o%L@>R4pn1UUq6<T=;EEosR<U?5$"
        "ypf;?J!c%{1H3OvKtj%X$zCwuOaszl-<R%O<Rxfi;-a&!+HF>))A+>~#so`5nlYfCDMhY?)ILImO}AxH_EM_>+GNMcCabCq@?M#Y"
        "<ob_o!xe1h$dZMhj?u;erF;nF{?>vR_h`}*-4Xgdsm~x<3b-5XM~AU*)Cy_xc?G(~q$#{5r!zV2z~jgbn|=q8i}zqa(qn*E=^?0s"
        "tl&ir3XFjhLMlXTu$!-O(;T=3lR3+FIE}(9#cUN$5iq(S6_*W-!>NWCw4i}XE{yjOHA=)1LkBO=K+Mre2x*;(TVp`Re+Vxgh!Aqq"
        "TYgL>%M%ooG5o^#tpu7{$hiyjB=R!uph{Z&gD+m(0#hb}Lc^#*n;bmRku~i=dUa*%xAcU+;56G0Qtl?>&de%Sv|NG`V<+<rJ6Y3S"
        "OX1;V0%C)hF!1;m1Iz)7cs=kK?)ES#=3AG7o_`#{6b0iCT~+4gFqxjh4i%Mlg-FU!HAK9z0UI=KWgbxRW)J>{rC?$ftuZK5$1UCJ"
        "z{j${S$K94kygjr(HMK=EhM#~MDz7%^gQq5$RSE563)l8fe;m_-(BDS8d!*h@m<O91HvZ^mq2{!S3rZ&gb=v4M!-7`4i<gmWf)U%"
        "gy}p0f-Y@kAY4j8h9ymmJhCHH(Ar3PR3ybQ#cP5g9XTY!{KhaPaGVY(43nUuwC0X@U>d?=Tg|##gLSumf^{<(FY}17zrs5FObn5<"
        "%n05+ir{I$GVPnxa%n+QQJ3$%rY+U|`r^j3fY4RZX%5Hog(=svc({q1v7Y@GMPIQ7Qd>;4YNRurr={#ANioyllPhyM^q7SHB5;YZ"
        "tmYtK21<=IHLh^Oh)H$wtB(PPUPUJIPk^PZSO6fN?jVg%r3V8Ob&T&((?xZgjosJ;QX)u%3mEs8H-Qy2B0E$lH!9G7Lq&RDHbr}H"
        "0GIrnDoBX|EwNQ@-?w8YjD?zo^HRmZrz{qE+fx!R1yE;0JyzPC+G4~=a0bkWq*hV_^#f4ifhB*$7@~mDP;pej{}k^cwu4Gx2sUSB"
        "v)hF0nA{(D&jYn8&YC0bf6IA?qQ`0uv_P(j)RNygs7zXww>*#oCU{WwZAM}$O&mqL8|v(VfEOtJM8YC0KSSr&N^Zn7dIF&Jyh8i="
        "iS<nr@5V+Y;&B1RZaWT++q2e0tV<*_NgQ+5sMcb#avq-pn{<ht&_w;Ts21=8JQ*vj69XH+e=1{eO_7=s<$i2O5ro(yLRYcOA*>xJ"
        "w<lCu>4D^?fT)aqZo2;g5<aTz1K<(-7%Q`6*ID7zdxIRFQp7EG*LE2|2BmZAdaU1}G=mazO19c=0G)JG{1nQva!6C+F(_Qe^Q%SI"
        "#@A~w^Ik|vv3!mvV5ZuWDFGPNN%(R`bbl~$n_q#`|A3`>axAC=f+!{UPv~x^^U6h7eZn4Ut-Ht6T9KydczN{B#$tPLDlEMUsT%<%"
        "H*~Nz>S#ZOq+Y6JlQ?_mh@qLWUWver)Ivk^;1n9CQSqcRpxgf~(1Ib3gsMc*+a`()#44EhrfhiD2h)s5wHes9%!I7&9SJy(*wcal"
        "#2>&&G4R0y>^DMHg^dD>*EBA|t0l&AWkT)VV5Ma$Cxy$!bKR4yRP`Q-M-u5%;HVKLBfIQD5CSm*qs~KY3}FNrdiW)kc7?)uEhf-&"
        "pWc#kRZ}h~{kRg3l8H}q=f=~D%xF;xt~b4J`$7bv<&@_4IL~r51~Rcc%y5l|DztRK2C-06o@5nNe@^ZyMjW(m>qLiea?>r-Vmy04"
        "2|a^$Dq)m6-Zgl>l8?RM$*R+p3(TqN`!u`tkP)v;JX|$X@nL#lsaAv)U%77*tKTf4t9Yx?UBF#JlwT7}`SquWk-KVg{)u~(dCiAI"
        "TNs?yHN6Nho3XJbF!UZUZuWq9KGtHaA(0B%8M}bzDyzG+E}P)oQeLSF7*nW2Ecr&gPbjb;sj<6XPW#B!P2H4P^R!QF&uo;>I-|T`"
        "n+ZHM;_QG&RIQF|ZVPF`d8#eZQJ=ee8R^am6^|Q!55;%~&VT8K+?~#EAeHnxmep8f;4bGl!OYoq+A^6%<Mc2rIVG!2OZ-gTAp|JZ"
        "b5<@#;LcLFX>BA45Sqkbm96vOKS1uR%cv8j`)3>*^z4UEe~HxjDUMO93WJmk-XGeklD#!3^OTCl{@it`ZK~H1I?2C9UW_SX5MIY|"
        "EYsZ^SryFoAqARN?Y<nudTZGy7VU-|3201DcX~;RC)&ouY|_~jt;7<CX{ECJ3)=PjAAcp;GJ3OiQ`VD!I1?WaBqy!R;_Bb9G-Aj)"
        "vdCvZ)sz*XMb&E%crozg(-t^S-<sgc?%5}a!zl&&>hi|J)TMKLt%OEf)uR~T7Z{Dxp0?bhr3e(qp4g{LY)ntI_vGP&Nv>zIeNwSv"
        "ACADl_xp5kN`o9G|EVJkf1Js)1|dLN!m*S+!B^XLD|-H}>psDjlaAY5_jI?7ALopS1aQimgFqbj<8XT7i~MnYg-wSmqX{qFsb-u?"
        "khQ}$PBgM^rPkb)cs@#CQfwN*%+6|}0FJnNT^lA8WQS?UnnR%8eTqx5u?Am@Xd~x%LPUcH=_*bv(kM%)-Vm_#{GJrgV8t>dS=x4b"
        "OMACy5lt{lw(9?xT7AkYNu+pvW!-jxg!3EoF3jU8<1*$D*eRxaQoB!<sU5LtVdCFvtDesDGN`RF=FEl-Bk-FqT&C2=h^ZJM#bmcF"
        "Jf}&%d~V9GR|tz7ik&3P-aw&qYQ>tRol+eMPixHrH9g>DmKKv%dYGiThDInmi+neVrZKr|m2H8=VK`qjlD&S_3d3GkN_U?EUoVhU"
        "(dk+$X(_DSmL@_mHtWYmZzbv$JK5fjW5MsroL^x@h<7LQZrsq;7;4pix}o2U;I{<lMR4(@zKJmw+2n^Ls6c*&F?PA>strz>Z!zJM"
        "p?2!H23}&%mk(*jv)i=hb=x1w;)Mn%*AU01a}z%=sn>3;QaX&vbRAqq>;nHjgTI$y#OTH6;f>o|lufd0Z0-c1iU1^9fMQ>33qnI7"
        "uWjA*-=7OM;@Aua4%PD6#U`2%q%4gPI{!RkU*gsmui)t`eDg)<2FJ&NUz)%#OyIAG^z$WPMe34^EtvpNFIi2zFfN7lgPH+_yb#NC"
        "Xk#$mX8A4oVzqv{N`oS*N-+eqM?Aj|bn3eAv!Vq3C1G|kK2jI5!`7n#2k(f;Q3Df2+Ya03c+{0aVMT?M)WcH9kzgOjI248idXJ!|"
        "eH9)}+hC?m9hX`H*~i%yPcQIA$i9Jedwn^ehe$d5PCP<SqHw@G5vMAo)sx!(X*d4J1K^=gG3HQ@mW{k~qC703sIi+OiSjZ|%FQSD"
        "k&vn2#V12`P-w}<v`Tp!dSlg=s8Wgi@|2I%>`bKi*ZbqhKWon1{rA+dCd~Ef%Y%&UStk{50vjXM%w9p`+=Igcl+zbCD`b+3JS|Vi"
        "8Z5n#xrDi3R{I2XufxMyj;iA^40GM&CsMS^#I#UQHBe$>?xZ5ARK!Gnd9g^b36d%mP4&t|z3?qbw@1t;ZR}rw1#@}3j|B8ML3%3#"
        "+};4U_knZF)5H#A5yqwz4_@&yU!edSVXMv_biA5oX}_rho>w_%=;~hKY2bR&lVD@N*0@2XZX+d5-eha;>ct^!*W23f>%PNsTdz+Z"
        "4}?Aok;RloRzh=*AfB>|$zyP1cuEaZ)PKlGu`dqL-D+|Rv>ocKzfJiA&&R=;vX;iJXK>SLu4>#Xzeg#S|BAqT6#X4k#X~>l1Cw`Z"
        "p~lCXiCh3Yy65^eQf6I71>5TIg1EpO7jWF|@P9E(_s3!SUD=Lv)wqXTOPy+`gp3ZwLw{W4XJRqgy~IFWPhTGB?_&FsSM#8G3e#Xx"
        "PeDhM2MhlXP)h>@6aWAK2mk;8Apomc;kJJV002!F001Qb0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!gmWpr|3X>Mg?E^v9hSX+<W"
        "G!%ZX)c;`2OH!t3yNFU1g0z&bRH>8-K_ygGk&`$xW|KJB&U6RFf9D)u5@#kmTO`D;v^x{~+|PF|p_IP78Cp>ZLwaV!o|9_FtLs6E"
        "-jG(ZTJ&4O`kJts4Th_SvTqMW^bJ?MuQ*X$8>xU&_8F!0{=NJ6L^nuf$d2h<E6VBn?^^cv?=?zxWMBr5h(CD?Un4@`SMB^w%^Gfa"
        "ucgwNeI95~wW1L%M?!nG6pgG{yX9TSvQE~#)!A%IVHE2%XdM`Y?}gzvrkF3D;pQFW8I+@F>%u)N?2R>25F>i<Vb<|ZszWg-SxzjR"
        "Wkq=~T%RQ#B!uV?$ST=4Vmm5cxaUDS{?@#4zG;!oLcQevJpw;^;MK-`ta$|oDze@mKxmH+@;9>Q{`A2_z84kGX*Jf2hG*t5@Z5$f"
        "D;v<nJbCJmjOY*f6>D|iv*WB*Qg+2wl=-vlv45;Nuk$Bg!?*Ogdjg4dHMT8rae$orV-$*$7r*{+`DjqQX~lMDz9GFNwfxwl`I<;Y"
        "UcCL4K$L56Ys1>MWYx9H(RJHNZZk1CJxOo#N+3BsNr`CI2!sYT4MA@bu!Y-Cjy(|I(W&@MXq)OuidOkBhEQf+hz042A}G;H15#vv"
        "dShnq?F??mAILyZ@(PaMNOL8}zJ95cR0;iFnq7!QO>DLigc8wmR8?F?H0eeSf^UR28BG_)o>*|+WhaWgW9`U;MojZofTdO(k|<tr"
        "L@etA-$<=kl{gd0x8(6zIdAyZ6@ox@Y!oJX!~vc<JUF~0?q8ZAQpjgHarg=GaaZvX#)a*fXjzHDi8!w7Wg)Usy?v0h#fwGBM~;Be"
        "1V90CGrXQG@cPz4w<VC*bu61FPYXQ`SetnLsfwa@7>`jwBQelZCN(Ek<ohd-iCRzkmB3|C4tb{qII*?O6&n1rnMzd04U{Nfo$O!F"
        "@U}LPOp-DW$jw^0Bvbh*urcPpsFBaf5NCJf6{a;;Ye<Y!eZGFHic3Gw3VX(z<@u?&qYL_448=#OYMoofjo?Kf3Z~oRBf1qAL^r%Q"
        "T*|pt2eFGgU*zh6DF&h9N>9ZI!aAFF4rYzwd_xM7$LR@Ox7c!1z`!B{@D`blRDk$}dS4Lq?xH3<AQulFeEIm!++H5z4}I0^v1wq!"
        "f;4yH-u-;}BWVyzu8ssa4(+I?qGEyS0cwY<CF_vB*3^#mBQ$$#oC*H%hZYn64k+&58c;em3|LKa@)z~lK%d{9n^w#@=%j!D$1w2<"
        "zNemme!x`tH0(HS|3}=!5y}drus0I_TyZn1-W_kqUo#!$Hv?CqgIogdTX8)gW^;6-g6dhgMy$g<Ba2#8CQ0MV-f3{he~J%%UKX}g"
        "*Kp?u+SExXL=EZ)b13$pS|jY$c+OkB3jySCkG{5(1I`~B6y33bKf2}zvS+QRZRW#F$6dT(b_TSJgEoq@4xF?X8ZQK%8I1*EUSo!0"
        "dPk4nzJ6JN7`3kDsCiK_2=&q7-$&E6ZcO1fA{d^al1&>?;)52g9<@*@dH5Yc_s<jUJ7=Mvl=RqbU`p*odw7K5a>Ecf^x$hHDLKf&"
        "+pOOr5FNVbiaC$AP(G%~^hk|>Wpxd`H@7=kX6NN3u~d%;{rTnP55Hx2LemT!U0dw9$r9w|bTrMwXEfAjSI0WoX?l{-;FBy`+@SA5"
        "p7o-ERnhZ}P>#DGXyT<Kr}?ZFTcCkcze{EixQnlzCP?E{a}6X-vmL*2x07Xkccy000}|w&;!xu*-Z`dRI$PqADujx3LU+tS>7|oP"
        "oq@MF^~nBD@T=9`6QTAL91S9=wSdNJqJ>f^<+aVq{RKTcy}p`OcRztmb5~^N{$et$k{!5Pl%fTMfptNt_jGYWKTK*bo;%&6s%7<e"
        "b#E$PXB7@%n0d?<!N$OI3kv&PIbYKwEYyB+j^_5TVe7m1j2E+?ATO~^^Y98K@b~N{ojxZ&;cF9%4#IE>&^uDXz5-nyK~6N(KR*Qn"
        "10$d|{OR@<+?wn=80Us;(wOy!ghHMvR$Nj<pG9GSBThoXcM$92x>yWXXDxgUnGwC?37{rWYh&e)93RG*e118TK&hTEuv#0^c}A~>"
        "abhK)C@hvHH^Ikh-?0j;HTY=iS<EMyd!v%=0aIn6(e-&wo?JXTS|CpPw*$!9+n9N%-S}h(FA;ou#zD&FJ~86}o*FXS{L(0b14eKj"
        "j!&a#6E_P0F1Cu*(NyKuu^`XkZc$New&HSGAZv&P?spnad?YOiexG=J5>O`%tr_~XR{G?1c6(Xx@$QG<1C)!OB4@boqMsa?3B$WP"
        "Q~6XW$7oF{tSB&XCs0e<#ATn*4dRg}E??6O&r=s9-<LrpZ$q|6!FQ_k1``IE24GF}^!A<SmMrtb#v;S!3?aFlIg(qezzN!VX|rgt"
        "#LqD}wb;ktsF%(%d{`?v918D47(|nk3w|}bV=VS#I=K00P>v^L-zT3(oytG@5KnsBR+hLk89prT#?Kc7$GzjZ_xd~`{sT};0|XQR"
        "000O8001EX6-Rh_n*{&>BoF`qBme*ab8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVa%E;^a%FB~WiD`el~-GD+cp$_50L*r=t<fXHBPcN"
        "GvER1;0^_vc1VJ~6oWuZw9Q2#HIho=729v$Iix62ikfr=3`jf|p3BF>Pf-+Iot`g=;o?9P6{W0C9vI*44WU}IdQ%<9Rw`nc29+IY"
        "I4hV=iIAW`er0t{$%Ynp8!1>8MbYum@e!{ZsSGhv75nj9CIq-8LZI4K$(n5VRu;6{p`)yjV#{}=v-s`$)4TT{);B3xKQ>J9nhA5$"
        "u;MG0M$y7(Z+s<7R()zs)0&jru)ogW|CX9!pORa7#{~bImDjAHT&*8z-BfJOYe@~dt||)Ad(Y15mu_{=tj4`{2E}7#2}SFJqD@mB"
        "@&~EPycgz;tax#F#!t=Ej+Hqz1`5EX7y{Wc#YDlbtisP;KIaOG!v*Y?$<3dvF#MkNK{Z&Fyiii>92_Xw9I!jbM@0n)Ah!UFVJD0z"
        "79xX(wqnWh_=v!-WLuKwuz|_*ShH%I5)F-7P8yezdx&V0YbgGbaAC-bEY2>y7yRiK4B{l~y(NQ;Z>`r10)a4~ADVmw)CJb~gG4nW"
        "ztgH^Yo(-$Bg|j`GwgtAYhv~ckliZM`8z3@^LtMfr^4tg3O2{AGTa_w))s_b^1w^8*RcF0TA#g*mgM{_CDHl<A1^Nd85FyH@{-q}"
        "?aV%h8D&_QKL&#?pc#WsGBvP4`fU(q7#4JE?ZFlg4M@XyQb2)2BQGCUr_;QBATPQMB7VVTRp)y~OXutZ7mTWSWF5NQaH?1DXr<ZI"
        "&GZ<7zh@3vJci2Ca-}m0qvYolx>P12!&%-}*tuBQ-pvS}_)eaprlJSHrY+Yk%xMin83x1JNk5=)lHI7i7tkAC=Uc`~Fnpk@TwRdw"
        "#yTY&ec77TX?6-Fw^L!ul{Twe)w0>x!UBFRw0M+2N)`w_x~W($;7WnH22)E*HQ(MRqi^-qsrO9wV@Zgzu|%}4lX`VIBe}cpg#KQd"
        "If|KAn%LtSZa8v0O;+05Di=^aSIq0Hn~(4ko{Izrv&&*jx=wZz*LG|<y6X~h@>6&dE~l`<^A|Sp#RC%CUHUaHFk)WnWO^`Ek>9aH"
        "?q@(v_1)^<v2ov3E1fTlPhA@d48%t}IyPNMer*V8@M3ogvYo;zo=1q=G@S$1^t69tW`nBV<6DYQ$sIR+^4++Jr<jQ+wfeqKI|+O#"
        "W?CgeP{huCI!6S0nG3q28z3Y};`jkVP_-gg{zT@w25J@0U;QP!=LKB*+Tf^&il&V$JWF19TmpUkBpA^L0e7?Eg$MX6B$)V+!X;?-"
        "z&%L=c_5{Mo^E-CgcP|N0hXo<shD<D6rDQWSyR!{D%UXo;>DY;9+qE0Ff$Yfb4W5HV;6R3z#LV?R(u{xI?pHLsL$hM5v3#sR<OYT"
        "Is7k@ZsGKYl)S;0#U;G`h%2v?@G!mX{LDKCSBux7<4gN;I(FL81=~<NPD3^uC$Z1*Y4#3lU}Nj%j$J20b_G<@X=D{32-z8aBOpTy"
        "&8;x=4fXV{p^DbbFomV9xHhrfBOBIZ58|DoB?na5CiL<Qm+gm2zXSv`^b729q8&uo6$Nx~Wg}%3JKS)+MIy@LAo?UhoY@bcstQKO"
        "oSBa{$xs(8F+{(J5%%dgKOq-(O!m5k@lbm%+ryL;vW9@Dy%ie;bi9DGAn*eUHzpY9tKNA}WxM15W!NQR{Xqr7YOgc_RSF;&VJ|>7"
        "U~V|_QK;w80-Ek26uCK)>pg8y!M3<BDT*>H3Xvadhsl6dh#<(Zjz{Wa#%K3ZtOPR$rO`E(b)UEZ0Vf^ZD4leiT1R5s@SQsEmjAa~"
        "JHsp)8jy-zXzO?YE07O3Zpm}#9u;Dkjk()VLjh@N!?i3cRNg>c_BB;^IeTP9Yo2Jq2h4v^O9KQH000080000X0D~B=k39we0J{(X"
        "03rYY0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FLG~nbZKs9E^v93SKDseMi71HR}9pLN-Ztl5+hLJB5|9vMJ^5O)GvWSFUgU#5XohC"
        "m$sFn|K6D;mp92tYrwX|ot^8MGsDSb^0O)|K3FJ)w$e&f$YKSeNa+`3mQ}`TVdWh{iit{5gTZWh@b((eSeXl}G%V@&LM~MzvR^UJ"
        "#XJZ;t<fM<R%WCV4WtSRWg%076oZq3SR|WR6$o@)=4FOC7M4ymmdZ#{yoE$0Yn%sHY*JWD=j5QbOlt8V$*L3qi^ez0k3&ebGN!u`"
        "x`1*L$>1L4YHiItm`o-?u+%DtR9KN@!Why>jwo%rN413AmYkO6@E4ib5dJ7xwa7yFRFU&R(43Ov$vOyvmz@sx7cNz0F}<YdE`&|i"
        "ivZ|{yr3tVvQ=0!2>u|$JMuSXSs)225T-8I!ff*#t(J)o{J<9nA26xduel44Ei@vLEXvA8GBu0Z#rY>JjM4*{He*pP%w~^ZW9O^N"
        "yv#S1wN{OdD8op~d{RI6g>T>-J`hlLE~Qx7^*3(?KL3>32JNmCx~;%Rk9`BCD$7h#qOICs0oJK9F72gOR2w?_4@rN?NR$axX|2*G"
        "D@%fNmQ)!*H1av(W{Ng<kr8(UFf(<c{6og_(nc8;3y{<n=Wu*>`s_*A3aqf`pFTZ4>CTEOk20}EZT#`+=#)&ZQKaqpv+kImN2$#H"
        "(aF=Z?l9g`e?=vEPH+^ROf4~kgP&Y?d?wZiVM*<7Zx-$43A0VkI3AKo*V#lXs>rs~uf|k4w_H^avFIF@RhIPx>*(e|tNV4$1g52e"
        "wM<hg;3JFxZx63S&jn;IA&L_Y0bQ()1hb52jY&@3q2|=gYxeryz<L{NNg}OIzCaO}wa|`4DNZ82R-fY^Bn3o{3<ptV-0~#V=h@H^"
        "rRT^*6-&9QwDYYO?kl}AU1G<8@tkh(uZr@4Ku=e84O5)2<^-e-Q|{@>rcbpF?`Ae1C~ZOZu2!Iy{0PY%Y6H$Qq;k20Lz*7UQnh}9"
        ";(l#?SbnJTn5skyDel5>jp?h0g)i^;b_d(;{sy-PcOt9MA$N4C1M_b+9_TYU#?>krBwWzJg&lxj@B=5zGp`v^)02AWy*WNReDm@6"
        "Y}PByCE7|Cqv~8WL(|AFJY))ZFE3z9t{bRmP+CHSsg_Im0JpbM9^c;1>Y3#Z>bnDdGzzN3Hen>>l2{=FOOkTq5M-4V8C+k!dJjY{"
        "4tDnnkT9|xXz~~_q>~+oFQ`@HCwt(&w&v#jW^O;rWe<Mv0*)A84R-kJFc(@#3i+B#_F8MDr<1E8P7UPL4iIzu6a}taU(6@7zC9v`"
        "X@{q9fR3;&dpNG=3|^dn)1Seqi}f8_A;4D!4Sp(P&rFA?x?&z8oWV8@*JUn<f$mwbhge<fxKA!NW>@%O5FdT@0n*1&gq_Oo;LrW!"
        "84H4xEO60O&H3H+b=|uR(gt~5(f!zBl8&5Va&OwJ{g1xvv&QGTJ_9#L0q$3x$E10WxkH#9ADuiQh5T>UkM^UV&xR(rUF=50YB;RX"
        "Ci|RO*6}5^4rT`L*Ql`zI<8mS6%#TK?ly3b@c$j@jz$;lBX%B**tPYZMA4IjeJ0wAZ8t|{!qa=&YC0TinmMya^CRxLu@%l0&{@#a"
        ">?q{#0m>(Xdw%1|<H=&IF&y3-$1vEK-2FXKj&$@EEEghO@*f|O>}98=R=PDvT6guIukL#qdD)9)Dm5ncjvDdIbsKTVGsMB|E*JLt"
        ";Z7##_vZRrZ_n!Qx`Q3-z0ez8T)JI$XUqc)OqtRGBx$JAzEUEAEcAZ#r<orqf<}Cr(pq!qu1dQFj=^s$rI7lbX|!V*KAywxwBi&Z"
        "$6iz>(K04r^K7Glu}Hg}ny2A&kn{qku%2V7G$PKBcIV2_Kx3`sMXQbmjbVRi-@2DYs_vP%jHJzE{F=FcbTx>rlSxu$+D}9XDZiMU"
        "*T=C^^tM+Q;=KMh`&#+iSL&!Cn23lrvO=%WcEx9M-hGqmVBpw{1Ri?uZ~$<xor71NaG+wid0(*_<E(XiAt_XC>OBkcH1HJJa46Ta"
        "_Iut>Ui)g9G9CIrJ}*L8v|iur@cXWG7vjyhFg+7tUqkROP)h>@6aWAK2mk;8Apph#QCr>=002l;001Na0047xV=rQ9W^ZF*Y-eR{"
        "ZDB8RWpHnEX>@OLd0%sMb!2%iaCy}{Yjfi^lHXI6|G?uXGLhuRWRi*Zv~`=rvp02@nW^z)>#7_V3nD>@V~SJ=(zY_?|9;&KfCNF1"
        "lAY{rol+BvB)ZXP^czk2zW**w^Q_=r#doF1>ayB<qHdDCw}U&ZVs%y)GjGicF28%MNW22R-f{28_rG`|E~~6qdq4j7n}zTDUw-lB"
        "7uj}KR<*~fH6ReYy|H0plV_{;`jaS&_M#N+SFwlhX;p5$xXg1N<Fvw44}UD1qUMzvuwyl%)DPdoB~n1oR#{rcEMN2OmM!A4NVByb"
        "clkf>FF(Bb`SR_@=<VNrzP!Q())9Z$@haQ$q8`CeS1hhuN<IL8vg-1IZFf1J;iY2W>#To$Rj{4dluoP(t0ijwP;107E6O5+A(2p!"
        "5wdO++}JvWA^7qO4?b7#e);g@W%S|lA8)SSynA~!lehn#)taO$fa2=5+;ARs=-?~tZla!7oF$Pcn<|De<-KTcx5H~DZX+ZTpTpyD"
        "l;pa+<wf?plqf}XUIP9**nu?TNu);LPUC+GPra5KGzCgefct+z+a7xJzdCq+A&JCEy(LN90_PC~xu7K>>jPfIWdd@%^qV@Jzw*OS"
        "udZ&Y!hu+^SR=pokq`dOOkJ&*;89#|w^@xh>jq>b=!;13)mJI}3B9zeJpATmg@VQ_;om44!j1*^RaSF*>2l6cEtM9A8(Gwg?QN1("
        "fxHmQk5vP5$saONN98SDh6aqpP23<pCCzpx0z?f!g=j!Dn257%`I_Yd5M@OIXqV@q_qFH$TKG81Vxo)iyhpi|F!5&+-))TVFFuu7"
        "VJ-&+)3F-82HFc4gIl3V8?I}xElMkrY|TY&3Z+`seO7NIo#l=ffnTls&|?BDVv*$hqN7-eo3fM(lmm@bj-7eF*s$}L-}>QV!ynW$"
        "gOrN@)?~1_g5O7FQ|}rVJy|NLA$E!&B%{uXz0?ZeWQD8`Lz6O=3GSiXUJ}q{Rh3oXzoRKU7!GzCEfC95srDY)G?szgW?t14fAQi?"
        "Spov)0b!mbY+(vW%IkR3X+3O?;6R$#@puLnG2U=3GGBo@;V`vxibn7~?{@VZ_HD~+hEjXdB9q8HkiGp3<_8<&ZOeq`-aXIO8=$_t"
        "FVHGT2Zq?lY;BtL8g@Npu(g|}buyb2uObbE3qX)9E3y<eqi<$91_nrd5Jj}7QK+QcJMnzE)Q;Rpqgo&$*ysr})rulfjl#=Q4v1s2"
        "CfCTmMFN7H@ahI+Jdk$?4c-AkFmHGqIO<*Bm>Z8?F|K(6>(X>xx05W0k5QJ~80N5*g{9m$Tw}%VefT>BMDLXYhXelvEQlXPdQCl4"
        "M;imphJtNjgj5gYgkD8!TT#llAo|8i@gp@ED5u@(Kf{l;`N42OGc|X`?=k`IZP^Xcx3mD$|HJ+y)}?Y>k}N3*t9{KyV2y)~)J8+v"
        "2Qu6}8{z%iQZeLWxGt&Q|6qB;m1ciL;yjW~LhWrcv1On~3*VYM=yv>s79T2eEGZ>IIz?~8jHU=6L~Knz$|0<@ABY8Zq;`F(y&|j_"
        "!55a{5XRyrZ3+Q*&qz`gY8;1gsHqv5@QSkb-j)TIgIds;7XjTP(Mw<=nVeKbs6FdJ!zmn(<l`eE7HL}n4*@GnTX9k5cRXmLnt_v$"
        "f7Ek<VIiQfBSlHjI;$n+u%0|G?*UDHUsktEdfL0C_o@KgV&-kK1Y!o%jZYjS=_S~jnfj-kIdBCdDR`tJeND6AAda93ybcU=he5~#"
        "AY|mSdC_b!PvL<btV+u9ThFV35D^CNuZmxQD9SU4+B`s>gRhpPhwT7*XerPNX1ncf)z#h_fPpd4J{BsgeI{yYEkotT@r#@l>xQj4"
        "YO=rE*BkH-N9i);6OkNG6Wa=m;eg`%HBE%yZrUa5XOU~|?g?8Ef!PsAsf2Y*O!GmUP&2TQU_NmhajTD-CV{XT5+yJ-1_s9@lD|rl"
        "g#$G@1+-kSNe-irAIrK1EuS!Q{0|@gPj`&CKU8WDaa2E4bD;+$(%TOoJ0SE@P#=a#QGc~yllAH>Oz7+#yfBfY(zmE+%@z)dr*~C$"
        "2b%x@qdBC5tB7qjKE2~nXL5(nR(ToUg5{-|muMxV#e`JMd;&?81DNF*?3-9>5E5N=6VkUKz?MBkDr=XvUP$Ao?~$}yaJFZcvB`2Q"
        "-4FDY1=kM3SRKHXy@wd9-Exey0)@X7EDMd1NsjLU&EZ)7XSWc`9QdLIC;f1M(^xY@Hb9KqV~{d}yYQ|*`$!*rOBY0_=M(>nnr^A@"
        "k21d|yP_u{46c2BoZd5#_G`b>IKblHz!r3zKS4uabKWvX<WLrD2bo}k`KZGfD}6Fz4u_tqQ7(d9bUI>L^2VAHh0c~Xjw6yJu<f)p"
        "&gDP8UV3L{9uf0h?FF6vo_X)C)R&JC?SUZE6=lX8zgde`QG+J<31SHW3AuTY3mGts3^cvS1v=sdgrh<4_RP2quEu3e<eZ%0-h*V<"
        "RY0o=uP<Jno;sz@DRfntqe-=kc-Ka-aC3dG_SLMfv=$-oYw(!`Nn!4zWJ#PLvGl-~`JT`3!1yS+lw-r0lG85OmeOKuiE~PcOUaT?"
        "RtW-NjaEm1Z5T=k_0rrpU#<)oH@R?wi#{$2w+sH5w^@t=$8jf(K+YVotIC)|tZxkg&BZGvZaoO;LoPtY<2ukX-x~qSR%2w3!D~BG"
        "hqBeRmW!K(fY1em=mMB>7D8Ghvv*l|k`r4qe^QljT&f&lKJR%R7&7UO{NEZ@v7*ijt`dt#Rh2I6KxZTcQAubovgwfvakL7*Tg?3`"
        "Sr?F-a3DQdLS<n9@ImeBEJiN`c^s&HzK^ho<J${o+pHj<2q79|k@VrlIt#RfO`bz)!m6l?99fe9!%U2=;S_PXL&@@g`S|)3WnDCV"
        "B#%EJP33Vzvu+v2IWngL_T&*HNAWEvMZ^l=6%Yr3kjy^yOpmtS8aiYgy+l$+q$8OrjnReADl2-mTKUvj-*hD4n`X-jRM#|n7-Zuk"
        "b^B7D7DXa$Pb)aE4+DG|=S@Nn4=TEf_($*?x|H_RQoz3?Uw7otM#k+-eA|52DAzGnYaUHk4U4<tsm#j(>Kd)rc31mY)}!lHvtI83"
        "Nlr3pI((Lb@fia%!cPaNjKl!eFbijaB;-(jV<Zxj=O?6eD?ql{vg-E4l*_rwa%YN=nP(rO_?qg)4(x(rQ8+WmFutNAuNpF=7`DC6"
        "9=v+PJ>ndSWX@62<f&9$dIG+Q6rMm9l{SThLjeJyiD}fr2=4GS5U}?3!CIyxLh;KyiBznLhD)x%5heI2X~sa>Ro6H~NHHH;P?I|&"
        "IFB7GniaYo3Zhy*S_AC>TFUg&H#gNpZfuS!lJrf2#Xpi*s+~aaWrtG!DqEvKb`rE>@vXfS|9ke$*>~Tbr{BF`uj1ADi`DAotFyD!"
        "cQ2BcjIGYT{f4J;`r^A+13<)R-<(QyoLS5`>rj4haTVG<a(TlQ{`}_SfGje$dhfJh{pwdc@A%pXTxkrJ+wH7So`Q`6g3kx9$m?eJ"
        "B;acT9IOUyv%Q!R+1NEJaM#bNG*FeHD=L_7U|2GMu=l9U-bthKLXPCE8}po`U#{Ji=1pXPN2jEtbpXSeK}<42gtdJx^LAuSz^Vs~"
        "RsbB?zvTjqB7n4|k2aZ)ZXxSu=5ocx<EeDn%!T@lb0k+*ZgmJ&&2%!6NI5sr!ZjLHb`aL~b*Wg-e9x0I7Oi8pKn^BdN+yhC=6Yb|"
        "R3JnrrBBHo2pHm<x{S*l$yR}SlBKDN(IS>Gps)H1o=VpcSmMF7fY5BU2c67wd_p@fHazj((;{CizF(ZZpiU<aRvLJ{C)JK~1_%<9"
        "2U@bmBJ|H{8wLzIirJ5@Fh0rHx+n$aNp=vge8KwnY^Ou(k;uG5(zZ8g{BczZfo_$K4#ccyd4FT_u10@UQ2hTQyl+np)Pe9)TC(4m"
        "dEzo`E9wg)a-NRgx4<-EMD_h7dQbrD#)&2cDWK^W>)$QsO4S<+=Tw`3A(<qdlTKA{t+gcC4`gurc@EP$d%!gp`v4Ski|J6eCOozP"
        "*52beASFzCNZ^+hi@^uYaAVX7J9QAiRpf{E!T#{@=!1rugMeiMTO2B0R~c%amM=QYga<hDmSKNZrzpL1@Nr&R`eU&Z3{5?GSotUL"
        "`iah|=lGuIoW`8vKgT<nr>E2ING<LE(jAqczi~#KYW<RIptA;UQTAfA|I$SzU>sRoM<hDbcrkFcPpbJn*)?fqO{}5kkjOtCQAk3$"
        "X=L`S{Ihg#+VMy{>bRQnAc)@^fUH0XbnOkyX_MET-zm$s<v}mQZhB~IFm9bA6jNdyww&Ww)TO56F=Uz$JunPgvu=W`cB|IokjcS8"
        "stn4bH6-)0JI{s%3uPS)4NcrtgEiA0#bC$;$EIqJP<Qp#)|_PtHD=Hml6mrRV>U+`C_C~|YA}rg3?a1}QG}C-VIQGM<WIU3Ya(r_"
        "S?#eYpm#(tqzsGyDn*rv^Q^le`?B<HNLs^@b@b}8M%UZtX--J7dTg113?QOIZ{DJrWWal4dsN$`*A%x!d0)t)$eW)!ezYesBo*>2"
        "gi0zu$I1vU*f&{ze9R|R&v>BqOU*o#&4XU{rxuwMxC}c&soTBv+A}Hl*rbOAb+*dsbp)1GJr~=a77eHjbYGFFaYMpYFO+V2b&@`4"
        "aD^zL5%_{bU3w)1l;q5eFhG&jqttMC#>7_j-fe_$FT6XFuG<-Wp~^6!SKW%ZmRoB7Q3g=v1Raa8eOQO$G=OT(UA>v)sg+uT1*)4s"
        "n?W2m$o50qq#9;Rb`A?L%(=ACb=jKKt})?oRHp00k{~W(wgWBHHr?n5H5oqL4}+K-;xHx0oge{9$qO8%%W~adM-NR8mMj4&r^M1T"
        "BUTMf?U?&~d*~HMT59Y$hNC7rY&zOq?KgO|Z`1+fD%;h5h%Uu<6^s?JomMaLEL=J=^$7OBb%L(VW>*}Oao5rkOBIc|7Ae~hDR^qn"
        ">C6Vj{u9?~WlKUNAbl$m6=<uQQzWaUfvKSG-;QZ|&Ba|-l?CR!zP<Kgp4J(?kq9VIf4ksB$I<=7v{lBjNPDu)yhqJ71fCl#Am3O="
        "_f`O@N>sDK2tD~H;+PIMyGPe%M1C`YfXdhx2eXIAKJ~k%+Lc1qNzJ!KNW)3bs;G;L=-^le5%kP=8xYk>(J4@*-i+(s7QsJ<ebJNr"
        "P!~(ve&{G4<nTXye}O?^!=)GE`<c3Q`y##{dLx|>sFwzh$qBMT_sAP>8tU?uMnhI_3<K>DMab0J0<NO}E*yTSD3l|NRw9C$StCT+"
        "mI*}!T!>ogx)f@1H_;-R_wn_gNLF$Je__Afal8z4fg+%<vy|RO)Gyo+ns)o$H8vuuyH;Z*6F?Y1t_pM<x_&lfmBCD-i(b88BD)@T"
        "CCHR|pbBlq1E(@5W*fuIz@#M$3_t8lH;5GYz>Zl?QZ034)`9iyr+Q`_rTfmyqx#O<@WokjneIqGqT!qxyY8n_fIpyhMtvE5gP9TF"
        "2k*_WQFuq<<_2=Ln4<U$Z`%slYJ|?g)Y+-i=hGln|8NHdj(YD%cQ013%@SMx-UyJE_hVavHC$cRAQW1SOlW=Yh=yV_dwz0@q}BO2"
        "$au%v9HnE#DlOK;u((cdKaGs0jk%CAGcp7vDKeTT%iSolR?-#0-Hqd$EcFL>DJvS|sFuqYyJbg0_oBDmkZ_1ovK3dBDhHKCIu^HX"
        "I$C82hiYG+X(`vQ76*eK)Ja_#<Qt~iy7CH#qAJ!{cMq!x3Cq%@;+9S>k5E*lSIbGIe~wZ<Pa%6*{Q*UE9g|xMNLyp}Tf2_Jd&;Lk"
        "x_d4NnZLG!Wp%Yb?s2fDKRIS~RryAa8C(^eL#GMT(bmLy0&xy_(*+#1x@tW!QkIsuK`#=EL1=KK&)FITW?gZf>>--~kBBZ7yF^i}"
        "M~;tm5RIo!$dbDyR(rem<SRifefmc>7JsYO4aRiu=}C}qnVw?`8bwJNN0E*Un-!i;(J<`n2Ue}*l|ZO^Nw$R*)aovea})CoTiM1+"
        "8R?4PC2-)=+m>_H<=Pue!<Zs4@aOYZv*(HhUcKM(rT$5P;iM{na?Glp`2B%{8I=W|;c<T!bFq}a$YA%Fhm+F}uwsOFH1R{4+On8|"
        ";sDG#t6e+;NnxPSbQSng5yT0;sA<&R4xoPSrVJ2CLFnGd1x$>zWa-ljWf3^0gBwOwnz?$7^jR3FX?-r6)~C27M~D-nlZ-%>#W*mF"
        "<n{BJr99DGp5ZApltF1Kzd8N>+XG0(fnS^-5@B{f0e*IRdU^l>B$IPw-2|3b3WZTMA64XeH^?7=9aif=cWGcBm_B)@#?2Aco<M$_"
        "c26L9vPMndI#$jmP&`T0+9srlt?D)P+HlWg=Q#Y6xjByNSZ0o6coHvV=h%>svYu!T8fup=sUD<t6kdlY4lgqTLxt+OY<C{Dn_`cp"
        "BwgAMCoxK6j;c6sOYW04-Hk>UT<2JDeFnU$-s?rBdC5#uc{&Mvtjy>j(Xrq9k#7zC1G_$oqO0;5)^Yf|XHUYZa-X?Tg8Ksm7E{O$"
        "L9623GohbsN=7j~$qpQ3?1T$F`~Lh87D!wnV7iy96HYWQ&kkalSji#KV^svRwJrytKZu8lrFUvf|3@@*3aU~Q>WA#*8+NVJ{mU=I"
        "jMyrW`#;F~=-;tEI=?*s0sjq->GT?oC?(1t`YEv#PRD+-Nc*3!qmS?Y>+-EUJax5$h5wZGeDeeT`xmc==dRt<d7QP8Fm#)#BP>~R"
        "v3{pz=K772nWGgh^#0vDJL*?O-WzAAt0q}vy1Q*e?X9>+DMEI~2S0)|_*tVP7u*k`hOGn>uWBAUgAq}W;CJ<qAHpt!7yIY4`aOUq"
        "jzOQq@OV%@pZK8w=T8GDGr}LTIyj$ssozyl)lZqGEX#Rv;e961$6wnY!%#oBcPh&*u*!%kXHi5ej3O+)M3G*kl$5KzsQLC1XQSE-"
        "Mn3-wP)h>@6aWAK2mk;8Apo`dk8j=u007w#000~S0047xV=rQ9W^ZF*Y-eR{ZDB8SV_{}yZ){{PaCwziZExE)5dIz@|AElHB!FMK"
        "Zfl1Y0hTmwkv2)Q+1)S<gF%xgn+qjsBxT1f^51tyN|xf-iRJ>0NKt(5^|`wwNv?SYk_nSLZ%g5o$=OUBaqAe*C4;*{E1^77xre#s"
        "@x3W?$*lr6NRs66;P4=|CS#I&E;M%z98+0gtZ%mH(BZ)^oAnWX@fGrui7nyy@F485vr=h!M&NG_4;cQGe8W(zxNx5P(zQ2CX$7=4"
        "iqaOw)jRa>?90XF`Sk4a^Se)DcEXP0xdx}&pkMOt#rH{%B`CRu3{YK3OEgTX=7VBQi?O=vXGVf%t^lD@C89ydQepw(wlHWXR#>xy"
        "oaX|Pc>i(lDd6*cz}|A<U4{J{pl-?=yY8V^I34b>8TVol;0p^<MfYvK+>ef!FLz*)dxpYj%@L=<y|LiUE#&GehKUD`dfI+!tWfzp"
        ";)6?`Q}Bm<VAa*Zj5VOjbA+042g8dAtHEsyNJP%K_Xyt6wm;IHh2{blMoX|8XfAo-@!AS@V!6smCwl3?v-*92MTc6=3&u;Sd`IM;"
        "l3r{5K!nzQl*RcVMr(g!0s8206oI9Hp>Hv*xK?vAjyKF$HCH*;(Ll_kDGKfCE1r|>)jo%~6}U|KMP-#sBM@Tz$EY86SwRl(OJ(6<"
        "tUVEGS1(@EO*#5_2~+f3xHzPd8^+xV>^dx%MT48HzXCO1c$R^kqsA@%c_!MfD4kks3n^HjWNhzN`0@5BMiiQ%j=|Qkk0GI*Nqxre"
        "dhElmry(4;TW1+~s{|$cKhYRZ*dB%2m2Z_0kj+3!jJ-M*$D7+%#$s-BizH2f;1)=`82Y;*LYB4E-2iO%>R3l7P{e4J10%xDW_leI"
        "n{jcw;#NLu<6UW0p<x<%g^~IotIw)*uD!~Of|3obYuE?KGR`z#W43k5tt-l{eemH=h5EkKy1E|K#}~HpIFk^T58|pD^sD{Qg4l7x"
        "n?_dk)PYWi!Wt}bpMDQ!wi_~M;hRp-lbv<KSG+Yevk-?%sH5m6<f_ThwXQ6{E|sH7ALPK#t-)j6a9{3Fd4n4speYw7RT^<_M(-!g"
        "6kcUS-a+g76Q+$RIzs>O6m${5@Tmen6~08e${-`l5+qhKbo?5(xI+ddvvmuxNsMq(z#z*`_lY2t^NI>)6Y8I<ZvwcjYbGj9olz(7"
        "i%t}#@@jWD2Xp}6#a|PS(eC>x785qd|KyrQ%8dzt8oWgTsaZpLLnyKWa@y}&NnSscsU4i#Rj!HpZ868#+Yy(B(%+Ab9dkJ^+`@Qj"
        "La~{Qh4|kR$q=bJho|WF{Ke1v+Z$Zo`I-q^_R(;ssM=L0*h*m?T0Pg?j3`8(^%X3+E`x`vk~b~Ze(~5}7(0EH*FL%lXP6Vlf_;mu"
        "fzr+J?nTD!Et_LBGQ3yG*oFQt$4$1{lYNfc1bXAVsY@SqD_5Zrlzm|q)c3deZHzRaa-mBZnyW~=k(H>$qY~7yYdPJ@iru-{I>0e6"
        "{E*1zQCb+IsjNDZDbzh0X+l|8f;LE&{qS*Qs!r04B0KLRj8*_(4M&}uz8;-=j+Y@tpxL;xtU23>Ek?x89bDNGp_vxkqg2FRFT_}T"
        "r1&`ge0_;^%r<rXsq`C1TWp!2nUGqHtrd5D$5UtzP>wGF{l;S(g7aaV;HGDPPA|@Jj{6oS)B%l$;fbb#>IsvlJ@nnjIWW$b?_(Q%"
        "Z9E!Xu1D4ydxWct#$aLZxEeokxIN87hmc37_LwxHj3?_U7{{?3If}EC&nyju_{x<+poh~^Gh62MkW5!fNZE8s9ny4qM14Uo$~~<{"
        "T|@xSl@1HkG)Zb<=oeKyVQq8#Re%3?bB-<^)cbh8e%v}yD--GCgry1YxM8&4hc6fBqd_LW`&NHF&}Iep==eTq-9wGV(eYrTQ}c>`"
        "X|nnH{!Q(+YD~Xp$_3NaEf~lF`xe!|XUX<O*F?$AL*ekE$Ktnz!hrb?P)h>@6aWAK2mk;8ApqM@2_lXF006xJ001Qb0047xV=rQ9"
        "W^ZF*Y-eR{ZDB8Ta$#w1X>MmPUtei%X>?y-E^v8m%`pzbFbsv^y;ARBd8ZdZ-4GXm4WY<&lVV8X%1M#9JpqZ0?G2y)hlmF;&vkb?"
        "o<VZ;7KnY>L)c=3Y9v|m0_Ia&1qf#4E%^rLFJFwRXoe{&KL$kXyS@vi2_^~36c?Wo4ZqN!)4jdP(-)Ams#RF2^3AbnDXw)**Vm`v"
        "9L4AyjBvp}1h0Mf22e`_1QY-O00;m803iTQ_x$jd1^@s*6aWAy0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FJpCbVRU6=Uvp-3"
        "E^v8`7~5{!HuOC}{((~;DhIB+3>yZx^V7Bheb@!quou@5Xo-r6N~AziPTU~>zH@kyNJ-8npf!1JJUn+M%d+37DNn2-b>B4aIw>1s"
        "<QFFRS8BKvgo=ukvh5mX*gMwA@<jgn_&2Ggf`7dt9py^DH;<AWsb&oqOmDI*yS=%+;cX|CAuTnh+nZX+7KqKM;YZ>ze_D*k9eFKF"
        "+B~qfrJITxx{-B_@(Nv>z|lxr6=koeVO4?3u8B7E*s!AJ;(@77abd0sOfD8nJTt{>4kA4TQ%f?8qFlf+ON!Op6cPAU_D(juyadit"
        "ikd(4iWLrCw6bDwuZq>QZ%h%`icZOzH%z}OTY-x-H&R_<?ZUN+2(4HtRfXoiy@9hx2RI5^1nUNox$aoGCTQ%U;=qO6>p`R{N(%9_"
        "+TPv}_`vl)OQHLg5!y5aH2um1`Sgn>To5>qsSiX*qA5fa2wxM$-f37GKoSXls2ll6DkZxPuVW(#<tq88of2d>kAghaZsm9y-g2XX"
        "9<loj@2*UC*k@LEjVniz9UMaiCIJo6_GQsh!E2^XIBy;!)-s!c38Oimn#HTageQVUEID^*tL<xt4E(Phf*-F*%rGKYqZ!G6mx8Sc"
        "{_lDGkL*p?TU$N9I0z=Vb<CP@!&>V<bi&5&^5E}?d2h)NAY*?RkthfFklk`deE3218}@{z0HVzUGkNAg*$P7<Psy#s?9H1u<de`w"
        "^(ChAXYmZo`?q`qy*<2q1~?Lagq+zDM}hUnqTKe#KsyG2EInnFZiBfi(DHgFEol0DcL(RJ$#A7@xbMq)PHppS4i2uLV?yLz<tc1<"
        "?_7}YGY`!U*JVvOu(%f<l^y0>SEFrE1fwoU3k_nFxPH~{X<Th+*D+BI!C$3OzsKm>U<goAvVvDO@(^e%Pa^6-z8Q-G`VPpk960PG"
        "LZJ-89sZ$B&wf=(syzGXidrd8SX+6;gyH5w$`geaRFSHOCiIYW(M{%+jZ*)4JSGEGp-Vz|WTC2IH@2k>%Xo3V&mvQ079KS`V05>M"
        "VX<*D_S8XTX_=+9E)%q6F@>*$TkowH6^yEfb7HH-lw)C{S~Dx|c?i30Qv4t*LKtk5diyfFsW05yNgZE0xSE({J{(Na^m!={Aquu+"
        "KgZygKgB_|CQl*w(Hgi7nFRf`=1&1?sDk2#yVbH>jMk^2O3rB(vTvzG`#=v>I%Rqpw<~YGmozRXAp~K%V8KA_r7LE7HKtf*2M2J^"
        "2&vc^_knMmIdEzU5f5?hed7iox3<F8p*^41al2Zly8G>&|FyHK-rf7{5y#v@LyQa{T9$p=H#S3e-w)EB3T;@YceYo0S!dc544U(0"
        "_rtf}dZ4CfHUlb#vrmB&S}Xu&w+BEaLN#O6?*2#EMZ?n{f)Uk=_kIY(8OmLR94}!c`~anfm0vmu&LHRSi0(Vg2u&c%VS<EN%4so9"
        "Pf#Ddl%L2ww$;%Z0TLS{z=Sl;0bv6kh^phfGJX<ZC_Z~I9Qz7_e<*@oLxlj3_Z>{-42KS{+?{<ZY)ZzPykR1@XRpY6l1KX9C8~zm"
        "#~p#Q-b6TDeQ+t^+Ri?%Ee+toMqUWq{(s$4b_QxCR6F7hbER{*W=&h(T+$&g)VxB9<`PXEs3m$l6qBIobbib9rEQr}yv#-(Th9%L"
        "VDyzJ9P$W7T072+j-2V}=U*ec>kp26wRd;%HE8+8lKYT#uwVmxsklc7y!$qw0L7PBc$Yx(XmBEy18arwT2bXGz%Z1+(?{N5k4@Vq"
        "1v)5nCDb2=%~Js3nI$iSrS_RL!^niwmS|B0qEb2|;i*xYe3f<}P{wu~mD}~EDE#$`BBfMRXzcia1qM%)r|^X1NXsuriGverjnuZY"
        "rp0{-trM?cB(cdd=<RVE=ekWqvc4ka43Ep>t3pPKsf-YZOPYAxf+neurHsH5lxD?{8BtxuFBu|S@PY<z5u-2|CnLj^R`?(D*T^+P"
        "9wyHB$z{{T4mG$)-5rV@pvO&25|=<AinlQtwit{__(}UcHi!nx<>c=cQK^c%Fn)0-#&7J#_}O=<Ys+!yCfOUo$1dw?ZiMoE+x<lF"
        "sDUrqB<bc?3^K9)Zf5(a(T~drANBlk8R6shAFU&bVG;<7SUOGwk+ZCsF+xLh?IyF%D)uIXp6`A3@n8)Y+MpffCap2$4YNZrxW@8K"
        "nxvog9MkR1zfem91QY-O00;m803iT{t-9qN0RRBy0RR9g0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FJx(RbZKmCVRUJ4ZZ2?n"
        "ZIQ7~12GJS_ml1&tW2#0R|cj-VuFqg&>f{Y=Nd@j!*L|~_V{`Q6{(qQ$MOI9k1=L<ci#g|5-8gU(F!u)_k<i0W_f^20wGPrf&&l("
        "<5vx=@%{*d22S8@yX%ZGZPPYOCkU}S%(-Vuun(S`jU%RHyW*_(6o>Or*rJun_(JlEb~rvW6VET(W<akBMM6|6#e0ub%Bn~m6ikyb"
        "Na+`;Zil%L-+-WX2jr`_Yq)*<SLoXY^eFna&J6v$cnmHG*ciq<8?Z^Xe3p%1QcF&xVDs>+1CG*EMQhGc%@XpsI5E#<8wy_CZ?!a>"
        "#MOUu1Lg^R$T|l;s&4Fq8hp9?#h1VyJt>Cz7jZ7A=pRqW9vQR4?hLVRn;%e10|XQR000O8001EXY;1#b7!Cjcvo8PuDF6Tfb8=%Z"
        "Vrgb?V_|G(Wo>O?FLZKYX>Mt5XD@AKZEtdUUvP47Vr4FHd9@l{Z`?NWJwW~gt3I6MtQFhMr3bfLz-gPq0Zkh;x!!|e2waJ~=2<H#"
        "B4vA9<iFp{kQAwpUE94o0f|@Q%y7PEI8qcvAGW+KS>0&4mA`W>>+0oizh1m!O<S+Tf>pI<`dZ*^(dD?zKD_<NbjxKW+VuIe=g;2N"
        "g;3|Lt-Gqo083V^ZdrXR+LAZfrsd7GO4&ysQgW?D%UYoUTQS^t3&^GV3*$ihmbQQXj2RrZ66?AZY{m1NJKh!xw(2x1YN3d$yySAr"
        "^18~qwiT6T-~aQ63w9&gN|a38i@ZZ>z+n_c$fn#jbqm7Ctq5=HHoqR;rB%f^>`_e~r|YgF9q<w%-Xfjn&wvr7*n8C1PxY@CKY?O+"
        "8v`^brYl8q4idpth&9VHSxKE`u@dDP<l#!3!$zsWS+JH@H|I=(FiUp&+F|0Qx<<5dlJ*o5zb4m&$i7{&lK=%TFq0DOH(qw)$F{B8"
        "I6@W;__k9Tbj+GsNl?B>qktwdm?D3`3I3T`Nn7!$ldVu_<06aS@=}R03EF8%8;({~&P!QsVDT(HVeCtAhvu5&EjyZnFYA_P7qAZc"
        "de2+F72r$pP+Vguv2FA&MiNHFsjRZK;JO1|7hs25nTsVY+zGk4)~R^}OM2Icr6IB|Yp%aK3*;G#Mhv}q3`4(*wpKCiSa&+$#qjvQ"
        ";0Sm?$4INXg@85=W>vOeN*BwM^aN48FG6f#R9RQNMHf+_$3a|bG{@VtG#=gmFtCQ7I9Y_|8Qi@I_=A}5+R8hRC6cfgW}{TsjxNcZ"
        "fD?pnVnPl1hKOIw4d^eU6qQjb0*IJW$$}j%z@tl<@6JK)76gcpnP?JR?D#uY$Xr{IA&3kbB^StHe>l8o-IZnZIZ(|ssy<pK91Ry+"
        "NCLAVlIY`$7#&BCMO-fJeioJ&ywRcnmM^cyj4qvQ0c&7YR)H%na2LovC`l=mHrh#XsmRQdij!oNE`YXFkeW5GauFM_g-=ik<5fZA"
        "(^fU5)Nz!8f=-jm<I^kB*^nGw(I>RizoW&>(#`&*+x<)R?9%7+DQLs7Fl0s_P<vVl6$CTdj0jU~1!W3wtmZaCXR2Enox}m<Bp|(*"
        "lmVwhbiQO}6Kmg;H;t%@m<W2PFWH6KZ6UF=GKc&qH`Nv^orLNn?jY1^CbjXkR&2ooO<m?Iu)D77)DuFD2~)n8Y!eCw{w=B;ezlng"
        "0*(A!S7KOW0?+V}p?8cd;DN@z{;};GZO<@xc;m7txv{?k!Hg05N8q~@2=0EkgO3zyU)&Z4Blzb!;KlF5?gX?TkcmJNFp~>Qo6Z7%"
        "k4#WT=QcV6_l(f^ut2qNCVUxmK!Ow=A{ozUU9bYa^_38)XF2mM1ElJ<vDZ!2ZAA$Rn<%x%PUdN@cA7TUqEdBhZsQ5kXP=vgfOMlM"
        "(L&a>+E+rL4?=Lc7Dm?^`YGsL%q?<hP>DrD<UEXt_ue`XhODdPr%pfz2}M>>vnpYYL(%+()hl#CCJP7~sdo@V5Hp1+x<<L47gPa}"
        "1tm$PLw7H%1c|E%$oj~~@lnc)BKFXuAUD#zSXxPXV;s2$;93~enF3bgp?eXo*KW_SRd9bl*0Dnmt>L+%GCQmoUKQB9_hg4a5c^sT"
        "08)CCnz|i!qzFz=No{+8hAD;C7q(&gXjExn6R&|y(`8$+t=nB}I;5E@ZYY>#2bE$=!M=ci%e!JwFRiMO#He+H(@C8J=Y>(=aulBi"
        "S&8=c5X8C_V*e`HcmHxD_)Yc*HVE!8FIJVv&Y}-Eqbbx#3QEne!pUk~JEaYLX{4YrS3^=@AJ4ZVw+NL!;B|`Z0A_FljgmBeCDbPU"
        "4%TK>8ScyPvqvyKr|Eerdn5>a8Oq~YmcD<5%2jubDar~DE;_%)Gr|$}@1jx5GwSKYG8!JOp;nhreTf8QJ+V9$$FC9}5170RQEWs@"
        "QQ0jLj0e26J0*-_9Cl*gveSdL@<Z1&<&J@i88$&Lt=WSbM6Qt|Xi5+uoGcLezJ>yCh~xY1V1j^c_ejjc4LA-V`mX!%D8lXEmqMd?"
        "g&oLcW2)qVmSfav5WJDy;i#+e2C^XPP$U{thG|Dcm@v_APxr%423#{9BP7nMgWY=LVx1kXBoFGb#P$--9I`3|cBR;IR$xm!R?js_"
        "Sj@dzjl*M;E$;2HLtUO+4HTK!{M<{&t6kjKhse79HKW@@_BkxtML%xxjv^Gb2H>gGI4B@x_wMEK<q%C6rW=fVT1_tszWrBB27zd7"
        "mxF*@Aj^X1*Gn@YIqU#tgDc+)jpPqRGzuVt#6G4h`&99kGZin$S87+~*KJ*4=P^k7BREWEI-}%g(;L=wHN>&gy5&rfww<5mtvSl#"
        "3@(M7vCeXcRGgYvkP<pLp@PG%s(7QWYdv_@<+)J!g!Jjo5xdbT7;_D(4Z7})h$*p+PbddkpPAF7Z+mc_L(z}nmFYUe=vW4)8h5!A"
        "hjtjHv$+jA-E>A&GOfwXiT46q6_C^4<xHticuv%Yk*g_vJ5~kqW~UU@ZInRwqyfXj47QC@!hr)QHW-9!9#oe4SjFzzlWCx@C@8fl"
        "Q&PAYblw7RL6iqWNq6;L;f~God;Srj)$uq4f1sd%18xM|MvzA{JO1)T^yp~nxSywlXV{r)VUH3xcz3IrQz6QYYMckQIe%5xWo%+m"
        "sx{8|1leFRyP0bHSBNc$wElow8T9>n#fui*$UQy)Bzo-=N+C3)xHw5+hiyDJ>OgZN=)x*#;hb#<iihDD(f(8r>`{?*_azI%Y_Ao<"
        "U~!I*Q?@<E3BaFhqjyrliD>h(dYr9-t)6JI5c6Tn;f2;SSkoM83*R7>qHV^tEdY7gP?%pO)MiRpO^Uap=E$^Tv=BtoeCongPtlqW"
        "`>8n-Zjuz25?iWmTpQfO3GN|t_GD&l>e@U=v&7aW<tu1wQXJ<b>^pYy-!g<PkZ<6;K5&eJMQxU=@VGCUIsZc6=!uO6B;!=QH!1t+"
        "1_A6z$jQ|E{%;lh5zQ1{k8P-E4l-&Y@fyuz9uR8^_=LHPLwRsEW5s{ed<L%|=%88bN=&Y!mjP&L-QhSfdVlfZ!+3rlQ7$$OA;1LZ"
        ")Hy(az(3<9>hhpbrKzbXt(U1WSg$B#X4=OO+w}>f-cjv<jb2#M*tNz6H~s@MTdduCEShKbx>lbj@`R2L5dGBkz_@Qe2@IJbnc+-!"
        "%nnV=(pfj3?Bdh;SmJ4@kq(+r2TwT1joPjQjLZtzJo(%>-r#`|c=Mz){Dis^d0iDmb$SqD$L^psMl%&0V}pK-m(BSLmd%gxGGA-*"
        "_i0X@u;)@X+SmOa0|m2z%)GNka7>_MNdJXKhHjOaFEms}-*1Qluqe&~w7~rStcRp0V_TDJ>{%HsR+i&a;NyzJpE%l$lWmS?{?T~b"
        "9BfE}(S~2L?}1t1s=lj~^H7|fZTSwQ!tt5FiNuzX-PkX`y!!^o_4<H*XU$j8KjEM<9nwBLG;o)Rf`j6^NhMS@U`5LuG7D&2W#FG1"
        "oGZ0rE$=f()f;^crGcK}j0rz!Fs~GKeFK5BfDkDPoQL25$}$s0(OU`q2>0}DlpULGEI}sjd9KSH$S>ds4gE`3q1d1%jvxB)ZBs&H"
        "OJDg|y$AUS^Usjc#5Ieu3R+o=*boY5po3K+G!$zuC46>@pW_VS@yY2|rzc;ZO`yH*hMV?}=?(iAfPeCOf*dH<PMHtR!DI!sTo&2H"
        "tScOw@^<H%KyeRwkWHj`0kR#Bv~^4c+q3J(QHIF3gD;b&JWYA&iVb;Wohkf64IDGGmdE$g{Ro#N2OeYHHS}KI`x!f8acs7KVJD=)"
        ";xRi-*vprbg(SH=C#0*v`;Yi2u*EqT1Nu{owm_o<;R0M*wLoM^^G;vkkiLJka9`IQ(XS0kOmA$)ud~3Ze}U7c9j(L3fGWl}P$BAu"
        "mB>)btktA2bw^)w?GPw+aL5$0>{{VN1YdF7(IhT>f&E6V>ztQ=7uzjA{_*S${tKI*5x}cA`0oI~*+T%YrdybmSA_hJkMm2f!lloH"
        "0XeDK4a9NuO%IZO8g6b~+;{+tOskYfg-&40Ox!~Om-q@-Ik&{;pj`UyCIC$ayb=&_f~P&dr*TKw63b_9os!ROm~Wg6uU{eWxA@;z"
        "XJ5bmn`!S&S0cj>-Msvlt!YbS@svyvitF7ic#nfT^2?JeIHRuc53+rEYHm->E%ucw;({;Whn*0ZXu$t3PtUI&7~qF&-~>ODRo-Hy"
        "?7zyW@6<V|{St%dDi|8O@1k&HviYb_z<%a_!D9O@%z|_!ip=J|1n~sEyT^$>u{}O_TS&}B(`swULLwT{g$I8<JtLREPm-wfQ+HBd"
        "$GyKtiTOTch&bwyhqwI^jM7YOzJKZ^b0Up!KzIAcsWTsT;T3u3hdz$>*#~I9F@kPTLwdFO9grMbd+>%*d$i&6eb~@AbiABmKY(WR"
        "hhOZ_k9}Y!0S6HQMM`J*jq?`2)4^sN9h+Q>s^<C3Yzi;@fR}c<bcbZB_2!&Bk3ueSc{Rh@`0zA2)Vqok4$bC1p0i*vdIq*1Q%d`p"
        "o}ssbP`w${)Uf!dyl!X=<U0ZC`&{nn-;&!GP^f7$G57=)mU=g+705E05B`LYLD0;-^0U|+43=S0ot|FkHG%uul-drJH8{eD?8EQn"
        "w%gh{1#XjIehW&>5#0n?A1``2UY3Y(>748;P~Xz2k5Upq(fcc;?yKE@08mQ<1QY-O00;m803iSxW4xPK0RRA$0ssIf0001Ua$_%I"
        "X=ZO@VQgn*ZEayMbaG*7ZfS03FK=RMWn*+{c4c!eaCvP}v2KGf5Z#gZhpQ*M0*j;@RXQ|ErjAjT2t_WKV9A`3ZKUw`g_xv4tvBC$"
        "_wL?3LdYRo<sh`Wpsb)1zHCy1dVuYY!`5gmys_XKReJ{pOZg7kI0tR-JfQIb0&x)vIUyvD;z+ffu^z_8vRVE(BCmTay)q~?2nYMA"
        "A&x3phvDbaj`C-T-kNTpaRlSj=AyjIXN`kIXKVsmd`tbOIvO!no)uQ|nu9_Q8%S4K7Do)yj|Djo>Z+axaI91dFB^D+(Q}Z0)YenY"
        "_nnq>X1hjo@|3R&HsLcAnr;xA^`~crauSY0_i|^gv6Sqtoh<!$^PmQ&&*UpP79g-1<FHax!y}fUdZQGau*cfd{lAUTbe=oy@$8>*"
        "fdgfWMWpX7-VM9Hr3_vn3xpZm^Z!El4T9I&2u~Te@@AB#s{|HDNmm8C2lT{Z%Xl`=pD<+f_5)B$0|XQR000O8001EXE#n{Uun+(M"
        "mp=diC;$Keb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@GNW^8G0WnXh<bS`jtwOe~{+_(|{KR~_%E1)>ZS<8>!T?;o`AWa;!z{Pj)"
        "<ysWmg}{}#yQ=l79=7*_eD}@_Nr{xW>veJ!a2Jb@;c)nyC$*mE{k80hL?rBo^7ak;`TfVUcq`(&s!WTPZADgzx?$_GX6;t6oVU=M"
        "un%R%SF9*ou`0{ECVctg<%>57uUb)`mPNK>1C569rs;A)^Q*KdV0u-Hw&p4R*_2%!3m8-j#_xFwbTa|;Zc29h!;Ix+A~IHWt1OLq"
        "o0i3l#n8mWqli0vXULhztFq=9<83PnYK=K?0AxJRL*7yx(8APe*S~Cev(3^~|Nft*Ec%z47fG4Fe6g;}oK?KVkxcdd1h1;6>(WFw"
        "<KN#Ji?wAP>v0WV3t+7&L6)i;o1(H3#dJtWQ-^`~B3h@JK-x5Pm9EQ}XB&~{JWP1Y!??^c4zu)}TT$_}eydtt(@6elu?edZM7z@y"
        "{g*EoJgY9vl1Pt<Hg(xm%}n-{Rh#DNucDUCQO=8WEt*z!cNK6XB1tyN1h0ui4UI@4z+Jq2k%%?JJw-{nfw?}4IJlA&WD}d~6AGIx"
        "zx?idKakxIX}hH}kds1`q3_izF94*mrmY3f6(xKSy79Iv?pRu|6aeaHJYOaJN(~L+e~Eu_e)%msVfarlV=K=K3=#uVq1>HM1QkDk"
        "<*qLDfUS5`TQYVUR~-mb2>+gEozSGyHA|bcXj)#xs%s`u6MiHhm^`ReXjGVp7Vzx{ArRJLaZ*dtxOGtZbSHLK16LrGJRm_q`wWGY"
        "qp;*!_}2Kr=L4%`Y2#XD;aI?;y*wm`uF8aiD7E`fQMQkUnugOLg=*GqG#Dubd$KlTn5%bSuNCU=x+?&<x+gAkKw`Q|v$Wl*wiDgg"
        "FA)KgU_R%bsfLj@p~TPqg@+oVHeHt`Dghx~Ef@e;n=-o>`p|JqnBPGcC_@dlubF?WQGDVN&Lk@D=v56~rv=!^xg>nX*51I}=_lIW"
        ";+suYu6PzTt*E|PUY&oReD?Z7qWK|F20)lc!_!8v_n^+#)OlUkW$mxMw}T^*v;GVxvhLulH>|uD^+OGZcEvss;m=;s3wk34HhnLI"
        "t+gr7PCfQYT3J9o=$ScQ2|+c%1fGL61#3HJp9V$?XuV?9NPob3g45V^W!Go|+Hl~68LJRigpCjzl4ziBP5=>Aa6td<QaB|rjy#cl"
        "N_BcW?hn&vXoP&Z<1^-Q=|Y>(a0Xt`Z)%s7Jh6A=^)U4eh3*?eI@SoyfFA@iV_t9DX@qxW;aiSCOBB_`20Eox+K3k4<rLo{SMAzu"
        "S*X#{dE#)#$hb?mMHL%t(h<)xKR|gJBhn`7;{)gv89;8aoLM54!%@(_lGcp%!j%z6Etd$ZR^ts$f};~130A5rN_DKl=57pq(K1|E"
        "jZ16qrlmmA4$Co|4(>EUz%@u3yt7CQ=mhrGL0A(;*?i6pnVU<`P+~2Hg*S5Wjkg5R?DP5t)`&MXN6B`8wjLj1nB@f_>3-CT$JWP|"
        "FzNED@ii0Rgdip@+W94TV^KiJ2T+}RUAsR0Q-5x}z7w@e!Og%zGk`q~<wni+-Fi23KxQ`Ch;<0Vc5koWy!+P^dnNvl*(sc@G;+Q5"
        "52dwdzf)!sb|bCxz7Z2^{SWks(omary5%*hEfY=5D+tfQ@jwt!ix^aE7g{D7)<OyTw6N`P(WvQW?R9}Opz#Hc*MRk5pNqh0z>YL9"
        "`TygFZC1uOq_ah%)A+t}UN#Oe(v&cm9CR-~@*ZU5zXFwmO#pc)pald4O&<Wmi=A&MgrPHvc~TjiB?&C>eQ9ZqCMU^ELxMdZOCbp="
        "@B}m17Nhta`XEWfNG(k$&tq$_^&$xeZI>I|Bj>Ssd0NyUoPH8O-q!}7c09q$p{?18g1J#;hV!(c#QBPmal`~zCS5vMVE3088c~{j"
        "e<3p*N|$SKpF&vDXWF4REH>T8V&q_!v=y`I*b@?ukoKoJ7?3m6s<XDd69wwzS+AWyxLBu~esiH=t-j6^O-^Rwkm!t_>ybtY-NF42"
        "yQIm0(FFERzW7X?-DKZ((5XVrW1~n%@MtLU#P>_B)Tcg<7M4d_dp&Vro|tXG=C9WvKyMOz$`A`*dfju_qnuY!R?3Ra+=I9j+8#u@"
        "*|trD^&vAY`D(miyqBm`1`ZXkMjYcU7oG_PZ)t&TFOW>ahaqP}%MCz7bDxCb&$#j!h5A$&RdeF5CI>I!*q@ZzlWpAb$wVPHq+|h7"
        "pgm!`qGj~ha}mTech)ZpSpGGwED_E2Y@CuD$Uw|P|1vQQD2>YwON%3z5&}&KjItbc?xfAM70@gguEKU|hJana%YJV;)<YU6WtXD2"
        "<i$q#X7CXN=+pqQ?3}%x>vdjBS<Bf6HT3TV`9VW1K>Fw{Yh{Qe#n6rqCz*$BB&@Du4Fry!S~~Jief`qMXiwV$cLqBH$il%rl=a=*"
        "BLk6uV~d3}4OF*7hb?N`3t;df+rvAt11m{F$5z7>KF6#p?y$@?!H&kcVXTK$bQNs07-pzck5XJ>7TbV23ZG&p`M!lfTCpw%S0S~H"
        "i8;BqiYPSAh$hI%d^R^ABL_0kMM+*{!t-<Yh))$_<vcRCDbH5!GZ+Q#<Q^iU^x!hdft9jbm4!il`x1m6u1X7yE-my!jRM4SP>a61"
        "uR1Jkh@nP02FcSJiVqU55m}vygbqUP?Jw`%D6~tXrEu3+z%!O=CY2Buf1`8@h7OsQz<MW6Mpv1Y4WycsUO`kxRs;<4$&9TKLo!!V"
        "{ONu5QRiHRE82=0xmt*e`288SJj2p81c^B7JUpMVbJbUyLFbY9gbmf?Vx(;aYIshX{lH}21y||?KI~f%Kj<ke<N{Y}p;rdIWe~Qd"
        "uLLqBU3P8NwOTSh&2tRnO(TG?93A2rVA6vg;ifFabUYvtJ0ZZyd?}u^fdm>f(l=-XuFVbzyajYY%&stagW)++z)QU36v!~7VZly6"
        "5}0#_3Ax-;nDLd!8dQVRi}Uj!gxtgr98>A+*{iv^*P*_OCIy1q6S;zGHbLgq+#EoagUp1D0YhJ-b5L@WoFsJ6t^G}lp{XDh!pkL?"
        "$r{MV>U$WQC>p(h^j(RETcT<u?(|FzgPePq8p$>qq&hX!pv}VWXY>B6jmq^pgWwre)lmwe9qa0bxPSKTL(U(`;Wu+QL)9j#GTuUi"
        "dL5A8H}N>`@-CCX`?spmy2zVfiqr2@#{;i(2!-V`x-rhu3PScepI?Oi!eB8XaWCrK{8Epjn)tX)h-7~8N4;6fiX#kp=cqd~ZKKK!"
        "bTwuhhG2b2Zg%uhpp2DuSAP$u?rTAJ@Mt6t8%%y9vJX^_!QK$gdo7)S=<gbowFOy4jewjuQ5JAgcyjatc#SAM2Xyx*Hd6;>8*1}t"
        "W`;y(h9m>_nq3T3!JJxWhjHlnF$EhPg=*<x;MKyvjTv{6?%@C(@d8Y#qN4#o%VCR&6X5Xt;MzBU0U%e1ylQuHfit8G-k+)17#662"
        "n=?sw&|1f>nZ>OEC?ZE!X!=r%S}=E_R=h2yTcxKjyGAvjTeED+c!q(~4l5=t>;yk{Xwx;1@j@)#mL<?D!~?ti<tDm$^Whq$b$k8u"
        "FCVU>_qRWOaMXOBb?s->B^I(GVbmg?s6s3);SGZiA7AMzEeybM<ogHH{fprea-_@XR#BY|c7}9Ohj7#qzDaogPwVb2tLK(#=o&?o"
        "h?wu@6vhC50({y0J<l2~QM`LH7VIkz&aWnP@HJvQIF@0ZLm$T)N!RoU_WcMxCnx)^od~)kKIxXK+JQ?K75?Fm_c5M7Db`)>FPo^+"
        "ib3MDv5jYf<kIUMB?jAb(5c7PpMpG4(Y#R$U{)%R{+f0f1a{VOOaJ)jpdcXHcHp~(%IA&x!xepXK^1)acJLg-0yOICZ&`1+(&Y$U"
        "Z;+ka6t7EA{EOvK2{eqCSpZXIkAqh%`Z42+2?9s$#&V;cLl7F|H=Z=Pg*kM$I+$(Cv73RkA%ngGVGVZNz}YMr_i@(5gV%Bs9{a`7"
        "<H)-B;k!Rue8*gN8WHfsh>~GfhC1VodH96Ap~Pj^hipK}EA`WhLF}X^VY{mY)nhUJH0!QL(h6iCZLI{qi*k_zZYh0j>^v$fH}w6("
        "@NI`iISDIMa(&q17$?K8hlrLuwHhRq3cF>@G^!%m)jakWS%<TWe&ft)L>Us43wyvoSdRg^ljZfZ0!w}8TtTh>lm*hy)Dl7VYBMsJ"
        "rqLFi>-S>mqe9s!s$+s*f@T`mWkgY`Fp-E4#v0T=B=cqn>!_!F(P`}wF{o7<6HoT^#dwjc$+A{S;IQCcNe<OwBPm<L33*kCBJn46"
        "i3J_@0?ojqykVt3>5D~4b{SaJB1keLowQ?S{$%1#L}gtBBR`WX`{JNarX~F+OI<gP6dIQpf|>`|Liu3T9fnFF{HN+bXh7OB!k~YA"
        "QIHE>P$A3Aoix1UImJ~s+1QA8T@~sl>@D;hQRI0}f#)lVJBJH9eP-q?>`aR~wCQ6CJ0@T3g!z)xnF$OCqgFh=gX}Bu$C&Sr>&dc^"
        "ld??>zyFp*R4Qx+-aJW#@_$u`uS0fYq&<|?P_J=D%L(iUaw?a_;;ZY_HbF>Vs7bBb;A;I`cX8N0!4GMJ-$0%f)06D!SjS0;v`!9+"
        "BJA2UYeLGgaRMdUk(Ju?`S<=b;Q%ag;slR}v8Wj*UM=S7%WvS`R*ZEG`7eIwF{io*mWi0}bo#H@MF0pf>j#vmS_K7)T`o-9a8#~H"
        "oRs>O-!xm+4W>ggsbk?~%6D58mXm&W!@N<>XbR@g7xTl%3^Qj)<TR?-<$t-kMUCrmIJ0f)DCwA-l@%w|<-;`n^p5*hl7{MR$qsOw"
        "8Ed4fw7v0}jG`~wMwMD=amGWfBcKAV`*g?iw9rWLU>>Q$$mqFq-6NuHe9UX2T@i2V5<kkBIx!IZ!jmzaEIJZM9Z?)RwL(3MAqh`+"
        "wxt#ctFrP(;u;66$cl|=Vut>5#pyPnes2^xQ7q>pCaCwwK(%KFClM2SZ_2RPZkEQUmt}Kj<j|8pLhvT*(@l*^%CF`w+Rp|M(V#R9"
        "92kC|gCB3CE6L>mF3hr29@mf4C<9Y`d2;fH^NA(U6^o?~b78UZ)wE6l)sJ?#=SX;+zv7YW(C>99qc)RiTnD^__wQo#5sbbh7}3$q"
        "BL??FaHy@jAZR7qM+4Ew-nG1@PG<ivRmP92s-vHM)g9~n_Vi(*zNhO@323^@V=6xS8HsAI<V@{w>)7;cMy0Cbt0k)Rd!)S?RQR_v"
        "Gj79hta&-`nUG5k2F#xi@h9PZ9JH03#zD|b;&KGRW6-u2Sx(-+9ANtLc%bUx7U*X$lYmHFHa>u@?QQ$jK6h>3OpZek-91GLP@FdI"
        "n?4nR{X>}P^v~wcSMZQJoZ`Lx@xxC)Tz|ZN=Y9FFW}b`!jbi!O7g?i3mPC+Bl5i|>SijL3kZbMP+^lDmO=;2EH(CN`Vb!>gCK4kr"
        "yxO&*v40bF$O}L1`R&xhy*!p}>-#5H*Ta6Z#n)Xy0W=)<3jF^Te_#;RT>%_Wmg+u_=(wL`eZ>XVvo5a={_sf&$B}_fYeZuaL0<4b"
        "P)h>@6aWAK2mk;8Apo>CALbJX003qj001KZ0047xV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPaBpmBV|gxcd9@i$Z{s%h9w7gLRWH)i"
        "qiI^ShwS(;i%o$6(l%&kb`ORjFczJ#BT*_zC6meezwdjbNRhN`r$ASiSmOKs=1Z1kSD*WaZ@4B~+VbX!^c}oD5h`j@GtFen1y`DH"
        "NY`uL@-Ix1JJ}7rBHK<9(P?(ub)U%ZKYb{(EIU6tKjUrRNli9g|Kwj<s_%m{-O0^;DFmU42!RJ%*|p@BZ@Ud`?pWK>QcDV~#a-F5"
        "wv$g)FS}btMglkZa^3y*?wSim<pQ{gEx#Kivv8HeM%w3RHQRzVTRJq_pCI6tH%#Tl<@p(bXVbN9C&-dK+jo4j2PPHoM0QCssx=cD"
        "UaF>(w6d0}9xk#)057Td1i%+Rh97Cu-&6Sd{deKltz_&As~$U9D>(ch2R3uuH*Dq*=qdc3{n2ruodi4k$7zPDjGxe1qldX5d8VG)"
        "mTAd1*#g9y-kdG)8T|}I1Z~+RdCnRMEWcx~e*mOMD%+v2lx96j|HIA(1VCC2Ol-QENu*yh`?J_h#QE0Avs^|J1OiM#tiXi)PPAbE"
        "Y>g5f7iA*hOtG@e*0Fc-dG!h-wX7NI@kef>pjgCEnZ*a{_D6bbqN<vNygZ;?if7dWaQh=TPJFp?R;)S!awLkJR$@{hCKQ6~EAbO;"
        "R1|k!-e>RClYy~|qf;_HNZ}QmGV}pB*Hp70MNNw3_RoT}z^_^U(h26m)ERj!Tjd?oxzn_;AXA`0vBJ=NJ{bIq#5TJi7C|J%$iHbb"
        "uq!D$nP(rxC(%6$;-q~&mQKs94ETq_xq4ENua-wj3V~b-(QLk3B%AU_W&0JxCkBpq76j|1tM3mNge#(k9?L1K4@n!9r~*;Yh**qL"
        "0fTB4!5tOzKS1}*&GnVHArXw#icq3hPbI_xsNuq*cKiZa^;6ICPz+Ok3oij0zze#vr)%<-TpSh47z$xYJ19+{?NA`0xaJR7+%&sm"
        "GL#~ph#97e3#F;p#G=_IP%KCeBo<`bbW|6`glgNVX_X~cTd;?A1Xd&RItEV;$=fCQ)&dl#^)pCbLCemxeW$6{kbUmi2JOEj`IPp%"
        "q=ZMALm1Qz$h4Uzuq()aLyJ8Jr9Gt?<ff&lz>uR<llk#uXy~y<Eui@oCS$w;NzA78&Hx9Z%}eqU3*Xo*vy!~=97x?cA8+o{LgCuy"
        "0u|Db3<|52^YN(Ex&$}1)69ZT1M`n9y5J#)Yc+(rTr84rQGE^q3M`pKFA|P|bv{NLX$~4`0x;>{d$!qT?}bTea{Zgp*-RI^G7-gO"
        "wiPwa=26$Te}&hGGs6#fnZDwfV0*cYL^%}X33o@R9D_b2h+;>FIDP&BL5X#Fp7G(}`j-{rH45SCxnzG1Trw=3x6mmX7>h@Sp<N9<"
        "(|#ibAuOOv$lM@6s!myJdMyeVE1@8tO3hj2v~hMZjUvgSB$LuaN|R|`=bw#>C|nyL;x?cGgIr=N$fo)3i%($c<v}kT@_0~zusEjm"
        "-QzFdB2&JvvDNCQ)B61G$U3=gjZ(ysAiij%nQ@>#o|ewul}^j{P8|}Ck_IRUsHB!wBBUGSg3Fj4XpGTac-e1GjF%04qXiTUw+x(k"
        "=(ZQLSY|j>K@?*vU&b>)P;&uiXw0R>#)f^5ODvc@u&aewq_@!2vKFjVaHYl3X1Uj|-z0-cG#?vXC!5|5ys0N6*>><Zm=t^`ISX!i"
        "tM%!N9FEOX*Bs}1#h9=S$hGLN4VAjWQLe>hf`VIkJ>bFt#Dnqko=S+N8+S~)TZ~t=0LGhUsL^V`2kMzB(;n$*qL9iAvDO-8$}=w#"
        "Phyt%<I+`5Fe$AQ4?>(l;~ZEV2zBnU3&R~0Kb=%|#-+CqYLmi`RDlPEmPs?<>PNm|QNAPXlGb$|aZS25d@Ug~rB;tz-+PY*)eUHl"
        "vz?8%bn;ox-MYy#bn^rN-Q_S-$H|;CQQOlzE6oMx9fpqm`OmCK6UK9OA$N-B7>|DoPZ!%EPn9<0nF-^g@VBVI#okw!<Gk^Ysc89D"
        "#ifd?`<RR?Bh<R2eGl5?c@pu?{eC~QTF2S(!o0oQiz0VLPKVC@NykZwSn2V60(=yH44>xVDNFa}l-b-v`Hr!k@y}J;LpPl~V{~#}"
        "iBoNyx|e(UTVCmA%=sr4;d$Q}8KP&GwsOrJ8!O<hWMwDR3Y?8@h%{Xv#H{e#pbHxU3sSud-K=e#-Ya~a1;dH5dLeAG{h1J=gA_H`"
        "*oF5xXecmCK4a}GjB5SVro#en))sLkKfb&9HC<+p7qd<Gz;LQ*=2Cfz(XmBa+^I(={kGDGpO2fpq)Cq?R>4JhZZ+sOgDV?DsoetC"
        "j9~7=9%SN}5&WoEvPpGx<<RDRiOs6X18<Ck#u#4jElz8^$r*gxq@NYsH2TtfK1^n}N5@<(s;%t4Fx<ww6pc1idkWlKUmZljRn@+g"
        "fKH{Kz=^miwF64^{zC<G-+d>uX+O4dYsPhYO)!g!%q_68{gp00sJCrBe2ODHSWn^P|Knn(y{e^xZ<*3#M7=zc#`1GCkm5?xaK~m9"
        "o$R~!{}zCM(O5%oM&2>=%rylw5nZ5iZ^;EgM$U3rW8gK}Bpj<22Hfq;L?;BhqtUL+@Js&h;zjpj*y1$~@d0bU#hEouU=T_DeU^*-"
        "qWpG2awFR}<f4Fkg<qU(9ZeIJ0GKo}TgWGl-h1Sg=MtJmyXnNmQGknRQAir_^Rxc}P)h>@6aWAK2mk;8ApmTEmJ5ji004;s001Tc"
        "0047xV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPaB^>EX>4h3XD)DgZIVH2+&~P4?*{(|;Zp(|yU?2tg&dL|N|&-ta}CN^;~g?Hay4={"
        "+h1SJrkj!ypM-6__w-3!*WG>iI6@*UtIz~{l<9y%un7AQsfR7Y{qFiJP|=W32|@uCvT6b>fd6`h${Mxrx~^^7#tSES44xgvnNq^u"
        "arR+`kQNq&o9BfjB+B}IA@|<@#i92cr*p@xnVQj-bw@ZIZ$wzCQ`=1B&w(H4RrjKp@c5t61}uisvpy(4Q7$8dtQ%02nclp+gsV?5"
        "1*c=%fIT+W?OOduI8d1PS_WPWN%&1NpwsXRHE=FNq3Ex^+z(E;6f3G@oCc$sAR4ypj(rVXMR#<X>`Z-QQ;-YAbSzTM0+I5pd*hr9"
        "cZ^0ySzsZ2Do|)aF~xX<O7BIIyr-JRR%jm+DM{(!?!fT*+YfNKpbHwv6m83y)s|S+Wb0FgM{8n^H1)4DzZ8O@(7Ow`s0Y+-pxfff"
        "H1R~S@}pT!K<d2+{aiym5pUUz)r#@&zCxb+?y_zE08mQ<1QY-O00;m803iVA%C(=s2mk;t9RL6;0001Ua$_%IX=ZO@VQgn*ZEayM"
        "baG*7ZfS03FLGsYZ*ysMZ*qBGb7pidaCxO!-EZSI5`SM{{|8q0BDGpy9JG7a1$+;8$O8A!V$ltX0zN}vWXdL@uOlhPZjk@IGyD=M"
        "Th4B)heRSd!<pfazZud=lKjH@PIA?W6Jf``6N;$rk+uAHs(9BD+HOfhRk>%|cYRm$@<cv=yeCyB2(4=(`H_@eTk+i>w18YBN%Hp1"
        "+c%Zy8q!m>ulWPvjn@1LK7HQ<uey@fJJvLGVRX20cfWnQ`{lns-Ti!DeExLzpU*QAO3+eCCv<PjW?S%xa&dR0O<%K1qFbuyqN}Q!"
        "x2%wravIXNZwNfS00n{-Y|9mIcZFbOC$=&-y|S;ln(I^ap=Lz|Jxuh1w+cG4Srfz<dTW7TbX&-75G5-__atXUqcyU^bqB%+CV0i!"
        "wy-0ZxTFIj$6K)SQWPq1FKE6hZV_Q69S~Sy+KZ;!vN|VQR?(qWg%c}!(N(-=60z|14Rkxqf!97JHnw(AAR0Nuo|QRaZ3%h>>a{Q5"
        "`o^N>4Ogr0Z+?95`)%7lmp|UZYJ4#Ic!eq#JEqb^bTvyd^394Q+Qh_;(eqL+8Tpmg1G^KV6KTQ@@Sy|mR=+^Bv|ZpJu2GU1EeKS0"
        "RAhw{t`qy2Y)r4Sn{OqIPy;|VwQURYKeZA$p<4h4gn-l9EU#SMi-BbsdV?KSLJ(Gi(P`#g4LbwpU>%wsElbBXe1H_xAX?J3^@&h2"
        "v>H-m%d3i1Lk(*O=-3gpXN0#>iJ{au9HeV?c39IF|7(+YhVM);0%;cze5~Q<#U*S1oOxgbm;p0eqm1H)NlAAMgTH<@NccdKK{Ao#"
        "#4J{;pvg@BvhfsA0rRaK#1^#g>r+wgDS&_h?@+~pn7T=FkMq?>3eK4&-LaM#@+{0OdQ}n9noZQ6w~yeD5iv?_fWX=MtBrsN1fEhU"
        "InWmP7`dxgM^OR$qv(<Tlg-N}AC*n~7&|UIG#z!(wLJ38ggDw4&QH59%!rHILhH+dC|%Qo{R;s&;eDFDu8vn;<N5=sS(}cgW*cOY"
        "@%t)~*kLxx58cpisW>^>FwJ1IpKLM}`LOW@0r`XhgXBDZt#7nZx7do~tYY{wZ|iqUJz#nB{;IhFGe_aEWS@-TK|=#x%gDcCdd9hN"
        "jPoob@5mAcMST6Gwip{a$a_`YJvl3cVTj8$LF&1OwvcG~HreDM5$8elavP;=%vG*F2#v7B?<xqWb9}h~Da)>?%g9AZp`Nm>-{x^5"
        "2ojj-IL~5EA5LnUZ*OjH{P@Gz=&WFDyITGuAM>9pTp-=oiV55tGj&fuA=C0CMMq)Fn5I>Nl07XSD`XQvpR66r8+@YOZ7~ZaWGZYD"
        "*0UAr)MGAQ0m}#R)f7}5mWeJr1<<;4AiIi_`#qOL3@z{x2>gHG0{Bl%D_)2K8q@G2WRaTfD3ml-{lWq2Z{(w3>~{u92hv;3cN_=|"
        "Qpp|*A6kk|N?#D`vPY5!2F5b76W!2*VqG|5U>8o~zo}$cW=XQA2L_xAbr$-EiA&Okkpq`@8dE~_^&lM{EH0(hrXdB3`22z}H8`jU"
        "usaM%0nJPRz*rb9tbknmLYoYD*MWfOYIVV2KgIf7W&_jC)Slzaxi0?G$sTZ=8f@9wS%Al!YY#SXmQGd^digHiGw}Vx^46w%bCk*5"
        "OEJQK4J3fk(<uW{);$F{3u2+fX%ZP6UcyO<L*0ip$Jqg9GCpL3*@}1iU<G`+$4@=eZ&Bcfxk^yijc2W<c>Eb~{-x?{w6mG<F|$Qq"
        "GZHc7#fv#_a*T=#s<XJsLIBGq)^0H=1OwQyWKe95WCCszhA)$~1WuGt`>x>dCsIA&@-z-<Ya@W0X40d1(DdfyCIi?mBd#7L0{+KS"
        "&H-n?bS#ai+w#3m1^6eA5=ii#2PwqAMeOYlUhq;h6zXp#)fE=d#}6-bjR^eN4qigdfhSid@$#>+&E!xNEWAECY540+kVT%CP^Bqp"
        "c~&oit1vW!jO?M;1F0Y#mnmdlW=atmh&fS9JX*s9Z0FPy>T*cFys5sLa5QXYNe7>xxViMW#h4%AF!8F6uON>=?;L@gpOs(b4t%m#"
        "Ddme<1D{u!10MspfHX|QELY&;G3fx|0CHY@`JvbAs5>~soeW?69r=b<3M&v4wgn{CHLg|)_U-eiH34h`f8n7sK%gyc3Ou}Hffs~6"
        "F`Q<P7a#__-?hAigGZo1msiEUtGC^tB5_+$8v6<pH&A2QHXP}Q*pzc;5)+=jA_)fu7{3;i)5TfU8Xlz+bAp*pYF?sn0BJ*q=I)?5"
        "+7r+}w?~p_8^FN7vMrL4T1`0s`ktLB#t2W}xu4oh>N{RSg)U%P)8snJ-pY~naS_B-NA-wN^(clKaWmPxF!$k$M3<3hWSc$m^a75V"
        "Evi6;@ifopTldjiqdacTyKwqPogE^e<Zi0*4WN#vfUea|E$!vLQ-$2q+aKOzlZJmXA&!^x@-(hy$NqC9VlcuhY=qp|z4jz!tj}Fs"
        "r%}Xna$H15?4v6Pitw;~T-fUneJ$Wui7VE~nK<gcXer}y;k}}D_9f=2brQO}(TsfNup1m-Foej<BQWJ&{djYLSL@+5O6B5cF3Q8Q"
        "&Y$J&8<1N+xE<(sR(yrf&|@JL>m{DJ-L*pBm3wHgpR_E8W~lY$-0J(>`67n~ycJ!uvUR^p2Ni92eToY(Pb7H%8r<swgJ?fIg}~Vx"
        ")lF-wM4p<4DZ$HRl+oOyI4c}T+$2xB^wp9H9VO$k%>E9sn*XUG&DO@3jVt6X{Z^6MUo1?UulbrD0`|Zm8=WROjf!#)04xCv`Z~3R"
        "fd@8O<Y``G#~u=9V{Tdl_ggsghvSUFn6mUs#N|}c$tImabF{yGR9sFqFY&gp(4PDb6y78UXq_T9MlC+N5;yP-Pl!=egZ2@`{w=&X"
        "HgGoo!fWRc*Wt(1%+dQ*<ok3V1n=X$^dC@50|XQR000O8001EX+W}19Bmw{c9|QmZBme*ab8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5"
        "XD@VeVQFq<axQRrg;Pyy+dvS#8~h&zof1%G)20_eN=ssw9AZLZ=uwP!XYHl?q3%aWLVtW`W!Wypp*l#b-M2II-n^-*YCZqhg4BUT"
        "f~VygWYEtPpf@OJ=Z&P$z!MQjFXC+d`0Z|WxBA{#*;G|EolGYxcncB}3WG6H1Z{gC(rIgi^q6)=Zw|)~{Cm8<(aH-nTe21#;hoak"
        "18(usQ_z-Ndg@7>!$*NiaiNDBOTv@`t$cbUl&^~rd?=Zwlds2`E;!5^x$AYvbUB?2dxibYkh`5ufFDB`oOatTV4@b3@tEK@th^(*"
        "9zJ`V)gt(AgGubVsNWc>-C5u^pw!HyLm{%w29x%#i;4R9?j4IS|3UQJIvqJu3o(VW2vUiI6dK#tcTtH@>?m}Ber2|UY+>WQVaO6q"
        "q`iWY6mT5vy1iq%sM5v$dgS)B|Fo+fT+mr`T2V~x(DfSj9d1v8*Wh#Nb1G5W(HfWV;mc6n*bTPZGMC1-%-Ab!r~o3FV%(u7-0)tH"
        "0WBp8u^itl%A0A4pS@@OnnI#PyPbcU-OjH7orL@O)$D%$8ElRTHUy#Ah8M~Q0^cWXxv61Rr~rFFvLzZ=?|39gVK!+LnX3jzWp)f^"
        "$2lY5LDU9T*%;WpvO0fNI7$tcFfubaGAjv=r1uNv7wN3ENBIXd&vI2m^`pcr<<{WXW6zwxOENV)ESBr~L}_4u5{Ju)*-|s6<hl3*"
        "P)h>@6aWAK2mk;8ApjsPk_V*@007%6000L7002x$Lq$$gMcrCkbKAxhe$Pz)!wzTKk!B#;ag(%3o@7go**cVml<Udk5?GL116X7i"
        "5XJubp3B}qQnEXJtH+aA5ccfZbNSA9&O(SU{JHLBxl`i3E>-8$x8FST>OYlr+H~U8{AD5jF1u0MgLw7w<@Zmo+6_bh^NSbz{eCWK"
        "qq(u$7Y%RbUVQrvExo^8{k9g@%Nz0LZh5n~U)(L%;_cl^d|claVs-mrb$9de4L&aDt((RAezkc05ubPw-_6C1s&%Ia(9q5OQ}{FU"
        "Cz}bklT9O9B|9;|i3ep{CuCQN(sUKC6Sc8obgB?m_108liI0mQ3tz6Z8?4@p_zNMOsBl+RiOoT*RmoDmgAJ@1w>$BZFg5(3;XPB1"
        "t?GtT)r>vStu+0?>g{e2X5T3*Kr2|Phl7ygurpTwn|2I>&Mg^sauBfZ)=F5}ZRsU%i-`_uD;sf3(ogglJ3J*FK?zxsoS+EU0^SXB"
        "47}l$)XMShVAsK#rV!Gq@Iyn27kDCkI(8K-D^1&)E{O7PioG6oEQq(9i?_y-CjDr81M!oL8(klLHS=O-bQ&iv^d+w~du0o7qXloF"
        "8oJ|u7h*7?l;C}QFNopakrk|HWhb`^y@)%zvD|sp3bEfQvH*kuZ7fOn#5#M8fdyhNH0VNJay#9ls9M)x(q7pTWncX8@=uqvp8*rG"
        "^`oS5a0A#LJps<L$_0@ibE7&iM5)1{6Y-g*x$A#7<4jz@YW$a-UFP10zi5~TU5zNsiabt&1oa3SYlpgk?yYu?!k<EpQ-Zv6ChFF("
        "VF@`2`PLo>U2j#btOWt|+nNmj88>N7rQv*%vMhR9cV#n@K_DwdX9m&eR-*#oP-p64kCEnRM{rIBP6)|Jl7bw5qX>{t>+NXi1#nNJ"
        "CYW-!`4>dRsrs@z@H6ns*igFE*0k`;awj{`Jmw2TNayfX8G?(RH2%k0NWnHDdByaQLCDcrA&q*C31&z=f7UGo1)KvOPfk4nH8}1A"
        "gS$gPoXoANv=qakpPv1bv7b+XVh;~VYl0_?yOa{T3r9%_!3OkaYh?xK@*s61H%-8V46X{qS&YU~`mmA-D+20)E#U14du%UwTa$6+"
        "Fktl|GX;Hvj0-q}dX#Msd&5e=eTaHqiLYMwy@I_TA-fv0zsyZ>qpW@aYd$D}R&lfASio)08^T*4h+;zo<u}ry7duMM3OB)IfIwmh"
        "!@bCDm>T<?E_XRXfY%175Rj}^J!tYN#x&T+=Z#QcXJf-pAjA7R&#oX1>!5NFG~@&c+nI)P1lH-T?qHi!-=89Ngs*x6357T|p|>DL"
        "ws#@P_Z7veR_c_ns+X2R25mzpX_ZyY0g|HoOvc+l*kHhPvQ?MC`5Ms4*0Q7mQshN6TJ}UOw2(4&?)^82^PUNu_55){#2n7IjArx+"
        "7^-bl1*J_~L-AaB#u7ytwh%3Y7oIvmk!8q$CBeYnO+<7zZU9a^o&<VAaYo9Mf<7;3CxZAB@)5mF_3+tJk`o$)YuX<}f1|*lHCW>r"
        "lJj+pK+NK)W<eNpf(X~JN;Qy<)&NKs=!%VOD4_Nh7j{HZ#?HGyU_wkRuadFQR)b54Lr!qTvt=s5S-uPWO<Dm;>IPRgKtMrUR?i}1"
        "atAl4)=i)tit?zixRg{v{VN;=SQnYoL{gSpzQ|!^5)7Hi(RRRz%h3@Jpe<X%P|rC(5e8;eOFafdOphA^6HeoLU5;jSkjkyJpAigg"
        "B2kf>DW|s`)dYx1^b}e5Y)B${&6Z$vDe}CX&(0<E5hIUB59#=2W|W%|F|$3|SL{Haje@`edZ`HCLGOH{EF+!zV}w9zaO2W|)w!G_"
        "KhAP7K+~(a_!S8l?))YiA#gCU9=Y`Tusz!a<w=)=wSpQdGAjwh6wsRpA2A0Yl5kQWgS{HSJTVdh4x4JPk&kqyyP^<qaBTeX3b>Ko"
        "V$Wp`vKbCnwN>zw1`7CKO2qfmYVSKp+%j}(umm!$$Cy3=zYOO4aRZCNmJqakBO&VJ6HuG$u%joQdgMLdq&0{LP4xO?&okAYz$W<V"
        "FPWb{NW|j*4<}u~I@J%DaL}m?fk=UhjytJK(erULCj!9-X*>C#hz&-Ss24YNjf4=YgK7Yk`M&@%#t!VChzy=qd-_Po6Ay=GU{{5G"
        "W#2d0teOs-Po_pZ^U9Tt)L?XeH$NYklBDD&j%eCJHajP+rev)x08(g#l#X>VM}iA?34OHbl&_Y6i9lLn8$j!h7sNAhhwN)HoPf#T"
        "R6W1P9{myO;9M+f41?4{IY2Lr@93IAZ&?YsmH2A{KHvLaq*|7kxHZmQk=5`dr5TZB@y`$dQZ#b!Mm^wZ8@1)?09Fg?XPn^(DxU*A"
        "Di*A|^Syc|s7x+8gp&oYw8Uva4)e8XNJd%}T3Mepp*2bQ=F4GVUR*UWU(ts#W~2+Wu7al_3ZscZ9JZKMfQIkqVx{sy)0}p1<sqTn"
        "F)#r(>cD@d5cLcjB+nqfgxy8}J_;ygJ@9{%D)zMR<{JM5x)rH^BCBLU)+)vBu1y1-6cc~|XFsPpd?7FSxFbZ}7Ij7yxkmv9Yd9wY"
        "S<aJFV~KxGPAjQ$95)}oaG8zUZnAB84wR@j_Fp&z;2Dv{aDtXYcNjC=dAeK!1il#IC=}cOE!jk|pySQUY;&syNVy^q?Q9M}9R(<L"
        "e8PP1xMvEE0&^}^&%#H0fpA$VWLQN`IVfU>lz{$#Jjytu(FxY#pTzPQR)Y{yuQEh{p#<Sc(Wsn*cF4*aALg{0ADgbaL^O-B;M+#@"
        "`fPc3zj$*ygM@q>$nKblej8-z`Br(pWx#MIbx!O{Zp&qb)>VQpWkoG&ia>SNxQJB}C&{@Sk9LF?d=@%I@io@X<(xP4S?f~RfsjT?"
        "hmF|eO2Ds42@DJhw*5IMDub5E;K@qU_;Sx`@k<W$lW?14`gEctbe+%#Yw9*t$kPpsEzX!)240xW&3s33#!~fh+7Qct_K{r%5^Z&b"
        "=R8CYbvW~bjvLvBlF-85@3?_OP&;Gm%<aT5xEGI;L+CtG?~6R<NFkqTgkp!u9Y8E8va0ak78|8JG;=XQdvB$$Ns5A9;J_P`V^J%H"
        "Q<ADuT{X6W<W3?dAPqN<!Ko)OM^+BA5HQ8r#G=_3G+WFS?06E0Y@Da>NX{B6^&-S_=~94M!qFrx_XtYyhso8(DW67?J7L9VNPjji"
        "JabLOa<c0wQ=ieN$a1KrUi0uo2a^x{VqTFz+&iBTCKXPu$R=ev@^l<k6Zgh|nof9PgLBk99Qox3>XQ7O1NXX#GrGC>*g=(b<U92U"
        "dzBg+Xp)&-$D~Qg;h1!0)3IzU_T(rmYY%YG<Eao6!%aTj{;#?^&&Wv8EOJ-|Q{5`weX@Y18E{p+E~3h_G2G8#8gHpJ#QH+Yxe>~Q"
        "Q<dVYC(Mc5Gkz;3AUt{nQ^sCu3;iG^!NF$>HFN4wl{xei9!H~CwUw5yvW{Dpc;)aTAY@=E4$(5>-jyLh9hjzPcTi}4f04%=C*GLI"
        "7H4sh_uwE+*$2Ei@IQiDKA!pI5Wqp{7&58Nwd#*i`}-;^_zqlyC7)ai%)qpGQI7h6t$~@8a7gcv)U4nX_+<7Qax1vc*NHQ=A(-p0"
        "b8({`wOx2`T#HY@R>3%jm=RIi&4D{$>fW(E%YcF0No`M>tQE;|9t~X5_5#(#(ZflPo4<~O(TSHYaZ(1wV0OJ0i}g&rzFsfZ(Zrt?"
        "_wVjL-iuGytJU@LesQ}NcdPsxfxEZjdilHf$6|Q{oJjK>^dk;lU3M5vSW{&q*pyE+gOvfh4$vl(xu})0XOI^xdcU}Te_Md@msgAB"
        "+tp(E>+Ns1%lkt7cDs7>4)nZ!y?DR4|D9s#?c#oUyXLPMu7i*d*DG+`$M@GO@!{j@!`=Fpt1;hGHh6;pN9bW&O}86#3&c(3QQUw7"
        "t?8{sK2L|PA<pm}io^uO*({c)n9c!l!V^ar*N!0HnNr72o6+81^U|m{zvn&ehC>woZ4OU@VR6}eEjPNMOZ)|vW&tFr8%PzF0*@M+"
        "qk@{yt><H#aCZigG~}a)PHh_vEUi?RMZ9z`CX?bcNBUyyUoe-zn}bGgi20FvTbyg83zD$ufS*G+x}`s#D~zd=x`G3q=($GI)_$-+"
        "E|#s_PHzfudH6aZeW&1hg?Ie<1vq3EuwcHjK$^o7W4r|PQUh?~7!NeZfw<+Hd#v)QmgBYcaidHo9V6<D{6u%&r5V^x<_;InUd#u*"
        "@Z^o*(A^qS?R7JmNPdQzZTeo~>=9YRh`QHOH>2f3C7XKe5&@zDdG6a3yllWY&TYb5D+d9D0f~hEc(xaW#DTr69yDFg*8Y1E$OdnS"
        "@TrBDAF|=6xwtN|SfD8a9^>ZMsgP$0@@a=$f114~mz2+5@dd(K?u_C2DveO5SN=551%g*ALKuKQ(ouG$;)C=&VD*T2pa@iLho8G-"
        "Gd4DE6Vw-G)A+F#ahn$iL`akQUI&hZx#Qb19bq*#CGX50`zP)~qtVE!ndEdt`p%|n@{5;9(fth!O^E$tMAw9^q%pCfbYYVLeHuGu"
        "@!%)U*lOvTQ5*B0)1T~9C;L>Yh8~BP0FSQD4Cke737P^GkCx1`d$cy)(D}hMAT>1g*y!?nwm6+iZw{V~WM?{{6_Q~i`QK+@neof0"
        "O^D^&<qcNdb6@QFH`1NctCz2Sgj}$r`8>#t__KJeYg5YRSJk#M*VOS!9|Tp#k*8lL7WpTL;sRfYpZ8t<LNWsTM4vJ!`Qd8dYr6Bz"
        "2L*bZ52OMnQp0C9G&PZv;Y|QVIiwEpOW!%Z2b#8A-(G>*e$m&Ev8O7F|D=iJWaE|74`*<&oWJlS*ZLocR$y3cy|q#kCBlU<l`UvP"
        ";dI@01b_s*hg9#5KOYPIa-wGL^k}fve*jQR0Rj{N6aWAK2mk;8Apq&FOCQz&004*$000;O0000000000004ji00000V{dL|X=igU"
        "Wq5F9a%pX4Zgg`lYIARHP)h*<6aW+e000O8001EXI@WR1U<3dFjSBz(9RL6T0000000000fB_5w003ieZf0p`b1!mbaBp*IbZ>Hb"
        "UvqSIWO*)Xb8l`?O928D02BZK00;m803iTfDLbZS0RRB)0ssIS00000000000001h0jCB40Ap`%W@%@0FLZKYX>KlMcwudDY-KKL"
        "b8l`?O928D02BZK00;m803iV9ixwt(0RRAr0ssIh00000000000001h0YwM^0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-Tc42H^"
        "a&LEYE^2dcZfsCX0Rj{N6aWAK2mk;8ApqrL91X$%000{S001KZ0000000000004ji2MPcHWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lN"
        "c`s5$K}1bOE^TB`O928D02BZK00;m803iT-E!Q^S0001s0RRAL00000000000001h0SXHM0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSI"
        "WO*-hZeeh9Xm50LFEL_fWMO7DHaIjkIWjO~HD+cwWi~TqGdVIdH8n6|F*q<aH8eRmWimKrHexenHa9jjVKQMfG&f^0H#uW9G%jj$"
        "Z*EXa0Rj{N6aWAK2mk;8AppFVcbeV+004^t003zK0000000000004jiaSQ+eWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3"
        "Z*+4nF=b+8F=RPoWHB-`V=*%`HDxnmIW{#lGi7EpH83$^G-NO{G-hIDVmM(qIWuKuF*ac^W;tPIWj8f2H!)*1E^2dcZcs}B0u%rg"
        "000080000X0Ozjx=HLJT0E+<t0BHaK00000000000DuA34FCXTcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8=WHd2i"
        "GGsV0FlA+CH8W#lIALKmGG#erGh{S1H#BB7Vlyx_Wn?j7G%#f|I5RRgF=l0BWjSUsI5#*oGC4LbYIARHP)h*<6aW+e000O8001EX"
        "E-TfY;Q#;tiva)tX#fBK0000000000fB`)Z003opVQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1yVBVPs@8F)}q}Vl*)_"
        "HaKQ6V>B{iHZVD4HZU<{Fl96~Ic7OHF)}t{He_L9GB_|aFgQ3dVqs!oF*#;3G%_w~b8l`?O928D02BZK00;m803iSth}c2j0001s"
        "0RRAL00000000000001h0i_TC0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFEuzeFk@n2Ff}nbW;J6qF*GqVHD+dJ"
        "HDoYkGBjp2VmM<lH#9RiH(@wAHaTQ4Wnp79Ic8!xWiw<sF=RM2IWB5*Z*EXa0Rj{N6aWAK2mk;8Appz=2w~s=004>s003zK00000"
        "00000004ji3labTWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3Z*+4nH8^5qWi(-7W@RxlIWS=|WMX7uFk@siI599bGG;k5"
        "WimKoV>dQAGc{&oW-&7|HfA?8F)=q{F*7z}Wn(xwE^2dcZcs}B0u%rg000080000X05d!75#Rs-0E+<t0BHaK00000000000Du8<"
        "6952ZcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8@HZd?YWjJMJGh;b2GC4M4GchtZVP!ZlGBz<YH)SzqIWuE2H)Sv|"
        "IAk+2WHmWBIAvjGWH(}EG-EI~Fl09_YIARHP)h*<6aW+e000O8001EXb&f<(-~a#siva)tX#fBK0000000000fC1SQ003opVQp}1"
        "WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1z~!HZ){mIWc5oVKrklGdM9hF=b&iH#an7Vq;-uI5aXjVmDziHZ(OcFk&|`Ib~sE"
        "Fk~?_H8x@~H)UjDVrDLCb8l`?O928D02BZK00;m803iSk$`O&^0001s0RRAL00000000000001h0YDZ20A+Y#ZE$R5b1!mbaBp*I"
        "bZ>HbUvqSIWO*-hZeeh9Xm50LFJ)w6IXN*mVl_2kWMpGBGB7b>WH~WoH!?FgFgZ9jH8^HvVlrl7I5jvhVP#`uH8V73H)LXAG%+?}"
        "H#jw9WG-rRZ*EXa0Rj{N6aWAK2mk;8Aplm-+vP(7000;d001fg0000000000004jirxySKWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lN"
        "c`tNwVQFq(a&LEYE^2dcZfsCX0Rj{N6aWAK2mk;8ApiutO~YmY003+N000~S0000000000004jiG8zB?b8=%ZVrgb?V_|G(Wo>O?"
        "FJE72ZfSI1UoLQYP)h*<6aW+e000O8001EX^CrSQlLY_(j1m9<8~^|S0000000000fC17P0047xV=rQ9W^ZF*Y-eR{ZDB88Uu|J&"
        "ZeL$6aCuNm0Rj{N6aWAK2mk;8AplujE6q;;003A3001HY0000000000004jipCAALb8=%ZVrgb?V_|G(Wo>O?FJWh8ZgekSUukY>"
        "bYEXCaCuNm0Rj{N6aWAK2mk;8Apm0VNhEUt005Z+0015U0000000000004jiE+GH_b8=%ZVrgb?V_|G(Wo>O?FJWh8ZgekfZ*Ooe"
        "aCuNm0Rj{N6aWAK2mk;8ApixHO*KdW002q=001KZ0000000000004ji-68-0b8=%ZVrgb?V_|G(Wo>O?FJW_YWpr~dUtei%X>?y-"
        "E^v8JO928D02BZK00;m803iUDqh(Cv0ssJE3IG5n00000000000001h0cIlr0CRF<FJftCZ)0I>XJu_|VJ~5Gb7gdMFK1<KZDBND"
        "WinzeYIARHP)h*<6aW+e000O8001EXxU6!B;{pHxCkp@oCjbBd0000000000fB}vs0047xV=rQ9W^ZF*Y-eR{ZDB8Ab8}^Mb1!FQ"
        "ZEayRUu866E^2dcZcs}B0u%rg000080000X0E6usgdGC_051vv051Rl00000000000Du9uDF6U-a$_%IX=ZO@VQgn*ZEayMVRLh3"
        "baO9sZ);_4X?kUHUu|J-X=Y_}bS`RhZ*EXa0Rj{N6aWAK2mk;8Appu<X{n9@000C7001EX0000000000004jiAT0m@b8=%ZVrgb?"
        "V_|G(Wo>O?FJowBV{344X>N3BZf7oVc~DCM0u%rg000080000X0Efb+seA<h01XZR02%-Q00000000000DuAQE&u>?a$_%IX=ZO@"
        "VQgn*ZEayMV{dL|X=g5Qc~DCM0u%rg000080000X03zr9XD<T)0CNcd0384T00000000000Du9ZGyni|a$_%IX=ZO@VQgn*ZEayM"
        "V{dMBa$#e1b1rasP)h*<6aW+e000O8001EXRRug!g8%>kvH$=8ApigX0000000000fB_3Q0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_H"
        "Utei%X>?y-E^v8JO928D02BZK00;m803iTpza8qk0ssK11pojc00000000000001h0nIo70CRF<FJftCZ)0I>XJu_|VJ~E1bYU-H"
        "WMOc0WpZ;aaCuNm0Rj{N6aWAK2mk;8ApoY)O~KFs003zN000~S0000000000004ji$2$N3b8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~5G"
        "bS`jtP)h*<6aW+e000O8001EX<JN{j4FdoG#|Z!cApigX0000000000fC14x0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HV{dG1VRUbD"
        "E^v8JO928D02BZK00;m803iTANXHGx1^@sN5&!@o00000000000001h0UbdA0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-JZ*FvDcyums"
        "c~DCM0u%rg000080000X0NxMFF|-%}05Vem03iSX00000000000Du7_NB{tHa$_%IX=ZO@VQgn*ZEayMWMOn+FJpCbVRUJ4ZZ2?n"
        "P)h*<6aW+e000O8001EXe)w6WHV6O!zZ?JnBme*a0000000000fB_a^0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HWNBe%Zf|pRX=8IP"
        "aCuNm0Rj{N6aWAK2mk;8ApkZ)ER!@9007fN001HY0000000000004jij%fe@b8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~KNY;0d}Z*Oca"
        "aCuNm0Rj{N6aWAK2mk;8AppME8Bu=?000jz001Na0000000000004ji0)GGib8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~lHW^8G0WnXh<"
        "bS`jtP)h*<6aW+e000O8001EX!AbBzO9ub|_7eaAAOHXW0000000000fC0md0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HaA9;~Xk~LQ"
        "aCuNm0Rj{N6aWAK2mk;8Apkp++D~2$008YO001KZ0000000000004jiOq2isb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~uJaBp*IbZ>Hb"
        "E^v8JO928D02BZK00;m803iU?@e8Ds8vp<SWB>pv00000000000001h0qdav0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-YWpHnEX>@OL"
        "d0%61a&UEXE^v8JO928D02BZK00;m803iVX7ro5)2mk=48~^|z00000000000001h0mr=n0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-Z"
        "cV%;JX>@2VaCuNm0Rj{N6aWAK2mk;8App;060@}j007k$001Qb0000000000004ji{l)+Qb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~!V"
        "Yh`X}dSP^FZ*DGdc~DCM0u%rg000080000X0Hp>uo9O@m0NVio04D$d00000000000DuAY&Hw;&a$_%IX=ZO@VQgn*ZEayMWp-g~"
        "bzyXAZ*DJNUukY>bYEXCaCuNm0Rj{N6aWAK2mk;8Apl&&;2mKH008tC001Ze0000000000004jiBhUZ<b8=%ZVrgb?V_|G(Wo>O?"
        "FJ*RNY;|FDX>V>XWp-g~Ut@1%WiD`eP)h*<6aW+e000O8001EXH>CH@8WaEk*GB*VBme*a0000000000fC0+a0047xV=rQ9W^ZF*"
        "Y-eR{ZDB8Ec42IFVRUJ4ZZBtZZ8&Q#aCuNm0Rj{N6aWAK2mk;8ApraYW(lqV008;|001Tc0000000000004jiChPzJb8=%ZVrgb?"
        "V_|G(Wo>O?FJ*RNY;|FDX>V>XXkl`0Wpi^baCuNm0Rj{N6aWAK2mk;8Apqv;?GB_1003|$001li0000000000004ji814W7b8=%Z"
        "Vrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XZeet3c4c31d30!RZZ2?nP)h*<6aW+e000O8001EXwiE8C*8u<k76bqQDgXcg0000000000"
        "fB^>j0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZB|hWn^h%bZKvHb1rasP)h*<6aW+e000O8001EX6y<DOKmY&$LjV8(CIA2c"
        "0000000000fB_}^0047xV=rQ9W^ZF*Y-eR{ZDB8IZf0e2Wo~0-FJE72ZfSI1UoLQYP)h*<6aW+e000O8001EXDv?LZ;0FKz<Qo70"
        "C;$Ke0000000000fB~ue0047xV=rQ9W^ZF*Y-eR{ZDB8IZf0e2Wo~0-FK1<LWpZJ3X>V>WaCuNm0Rj{N6aWAK2mk;8Apmvagc~UV"
        "0086y001KZ0000000000004ji&jbMgb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVUtei%X>?y-E^v8JO928D02BZK00;m803iT4l9bqU"
        "2LJ#*7XSbu00000000000001h0XYT%0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJf|OWM^eAaCuNm0Rj{N6aWAK2mk;8Api+!qYA$P"
        "002V+001BW0000000000004ji>kR<_b8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVV_{=xWiD`eP)h*<6aW+e000O8001EX$VPJ(RRjP4"
        "r3(N6E&u=k0000000000fC1<a0RVGyV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!3PWn*h_Z)t9HUvzR|ZgXa3axQRrP)h*<6aW+e000O8"
        "001EX`6tF3a037Ug9!itF8}}l0000000000fB}gV0RVGyV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!3WWMyAxa$#_2UvP47YGq?|Z*neh"
        "c~DCM0u%rg000080000X0P5!>`^W+S03`<i03`qb00000000000Du8T7y$rta$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MVZDM6)WNB_^"
        "b1rasP)h*<6aW+e000O8001EXv;=<25(5AL0tf&ADF6Tf0000000000fB{h(0RVGyV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!LbX>@5}"
        "Y-xI7bZKvHE^v8JO928D02BZK00;m803iUzq>>%S1^@ty761S%00000000000001h0jeGW0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~"
        "FKuOVXJuqxVRUq5ZggpHZZ2?nP)h*<6aW+e000O8001EX9+RSUu>t@9g9ZQqAOHXW0000000000fC0KC0RVGyV=rQ9W^ZF*Y-eR{"
        "ZDB8MZ)9a`b1!XgWMynFaCuNm0Rj{N6aWAK2mk;8ApmVg4Cw$5005*j001EX0000000000004jisVD&eb8=%ZVrgb?V_|G(Wo>O?"
        "FKusRWo&aVZeet3c4aPbc~DCM0u%rg000080000X0IONywtoiz08JMF03`qb00000000000DuALHvs^1a$_%IX=ZO@VQgn*ZEayM"
        "ZEs{{Y;!Mga%FUKVQFq<WG--dP)h*<6aW+e000O8001EX6-Rh_n*{&>BoF`qBme*a0000000000fB~vM0RVGyV=rQ9W^ZF*Y-eR{"
        "ZDB8MZ)9a`b1!mbW@U0^ZewLGaCuNm0Rj{N6aWAK2mk;8ApnCIu8%zi006rX001HY0000000000004jiheZJZb8=%ZVrgb?V_|G("
        "Wo>O?FKusRWo&aVa&L8XX>MmOaCuNm0Rj{N6aWAK2mk;8Apph#QCr>=002l;001Na0000000000004ji1WW+{b8=%ZVrgb?V_|G("
        "Wo>O?FLGsYZ*ysMZ*qBGb98lNc`k5yP)h*<6aW+e000O8001EXwfc{5-UR>v*$@B#8~^|S0000000000fB__70RVGyV=rQ9W^ZF*"
        "Y-eR{ZDB8SV_{}yZ){{PaCuNm0Rj{N6aWAK2mk;8ApqM@2_lXF006xJ001Qb0000000000004jiJZAv_b8=%ZVrgb?V_|G(Wo>O?"
        "FLZKYX>Mt5XD?r0X>MtBUtcb8c~DCM0u%rg000080000X08aP(@RkMu06!D}04M+e00000000000Du7vXaN9oa$_%IX=ZO@VQgn*"
        "ZEayMbaG*7ZfS03FJpCbVRU6=Uvp-3E^v8JO928D02BZK00;m803iT{t-9qN0RRBy0RR9g00000000000001h0qJf50CRF<FJftC"
        "Z)0I>XJu_|VJ~!YVQFq@Zf7rKX>)XGY;0k4X>V>WaCuNm0Rj{N6aWAK2mk;8ApmS_gL4=T006Tu001cf0000000000004jiPH+JL"
        "b8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@AKZEtdUUvP47Vr4FHc~DCM0u%rg000080000X02*Vwn^*w=0F(j%04D$d0000000000"
        "0Du9legOb;a$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FK=RMWn*+{c4c!eaCuNm0Rj{N6aWAK2mk;8ApkAoAMLOZ005Ui001Ze00000"
        "00000004jiN`L_Xb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@GNW^8G0WnXh<bS`jtP)h*<6aW+e000O8001EXv^F2+69@nRW*h(j"
        "BLDyZ0000000000fB`^}0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPaBpmBV|gxcc~DCM0u%rg000080000X0BnGk3yA>$0Eq$s"
        "044wc00000000000Du9MnE?QEa$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FK}{iW@&6`Zf7oVc~DCM0u%rg000080000X0O-oKpTGzJ"
        "052T?04o3h00000000000Du8un*jiGa$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FLGsYZ*ysMZ*qBGb7pidaCuNm0Rj{N6aWAK2mk;8"
        "ApqL}Oy48|0018Z001Na0000000000004jiYNP=Gb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@VeVQFq<axQRrP)h*<6aW+e000O8"
        "001EXAT5#yr49f9+bRG62LJ#70000000000fC12^0RT)%Lq$$gMNmrx1^@s607?K#0RI^P0GhP{0000"
    )
    bundle_sha = "ab7f6e2a39c62f7ea25d9f310ca4fd5fe9d345de2f0ca879201684336e6fc7b3"
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
    PREPARE_FULL_POOL,
    RUN_TRAINING,
    TRAINING_DATA_MODE,
    CURATED_POOL_DIR,
    FOUNDATION_POOL_DIR,
    CORPUS_KIND,
    REPOSITORY_SOURCE,
    REPOSITORY_TARGET_COUNT,
    REPOSITORY_PER_REPO_LIMIT,
    REPOSITORY_DISK_GIB,
    TOKENIZER_CACHE,
    CONTEXT_LIMIT,
    TRAIN_DECODER_LIMIT,
    Path,
    runtime_root,
    get_experiment,
    fetch_tokenizer_assets,
    hf_token,
    json,
    mo,
):
    from bifocalgemma.data.full_pool import prepare_full_corpus
    from bifocalgemma.data.offline_sft import load_curated_pool, validate_curated_pool_limits
    from bifocalgemma.data.repository_corpus import prepare_repository_corpus
    from bifocalgemma.training.curated_sft import fit_curated_sft, prepare_curated_training

    full_pool_status = {"status": "DISABLED"}
    if PREPARE_FULL_POOL or (RUN_TRAINING and TRAINING_DATA_MODE == "curated_pool"):
        if CORPUS_KIND not in {"instruction", "repository"}:
            raise ValueError("Choose instruction or repository corpus kind.")
        for _name in ("BifocalGemma-E2B-E2B", "BifocalGemma-E4B-E4B"):
            fetch_tokenizer_assets(get_experiment(_name), Path(TOKENIZER_CACHE), token=hf_token)
        _config = json.loads(
            (runtime_root / "configs/repository_study.json").read_text(encoding="utf-8")
        )
        _foundation = FOUNDATION_POOL_DIR if CORPUS_KIND == "repository" else CURATED_POOL_DIR
        if not Path(_foundation).exists():
            prepare_full_corpus(
                "outputs/full_corpus_raw",
                _foundation,
                TOKENIZER_CACHE,
                _config,
                token=hf_token,
                encoder_limit=CONTEXT_LIMIT,
                decoder_limit=TRAIN_DECODER_LIMIT,
            )
        if not Path(CURATED_POOL_DIR).exists():
            prepare_repository_corpus(
                "outputs/full_corpus_raw",
                CURATED_POOL_DIR,
                FOUNDATION_POOL_DIR,
                TOKENIZER_CACHE,
                _config,
                source=REPOSITORY_SOURCE,
                token=hf_token,
                target_count=REPOSITORY_TARGET_COUNT,
                per_repo_limit=REPOSITORY_PER_REPO_LIMIT,
                context_limit=CONTEXT_LIMIT,
                decoder_limit=TRAIN_DECODER_LIMIT,
                disk_budget_gib=REPOSITORY_DISK_GIB,
                progress=lambda _row: print(json.dumps(_row), flush=True),
            )
        _rows, _manifest = load_curated_pool(CURATED_POOL_DIR)
        _kind = "repository" if "snapshot_policy" in _manifest["report"] else "instruction"
        if _kind != CORPUS_KIND:
            raise ValueError("Selected pool directory belongs to another corpus kind.")
        _limits = validate_curated_pool_limits(
            _manifest, encoder_limit=CONTEXT_LIMIT, decoder_limit=TRAIN_DECODER_LIMIT
        )
        full_pool_status = {
            "status": "READY",
            "limits": _limits,
            "train": len(_rows["train"]),
            "dev": len(_rows["dev"]),
            "fingerprint": _manifest["fingerprint"],
        }
    mo.vstack(
        [
            mo.md("## Full corpus (all six configurations)"),
            mo.ui.table([full_pool_status], selection=None),
        ]
    )
    return (full_pool_status, prepare_curated_training, fit_curated_sft)


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
    TRAINING_DATA_MODE,
    CURATED_POOL_DIR,
    full_pool_status,
    prepare_curated_training,
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
    if RUN_TRAINING and TRAINING_DATA_MODE == "curated_pool":
        if full_pool_status["status"] != "READY":
            raise ValueError("Full corpus is not ready.")
        train_data, dev_data, training_manifest = prepare_curated_training(
            get_experiment(EXPERIMENT), CURATED_POOL_DIR, TOKENIZER_CACHE
        )
    elif RUN_TRAINING and TRAINING_DATA_MODE == "repository_tasks":
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
    training_manifest,
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
    RESUME_FROM,
    TRAINING_DATA_MODE,
    fit_curated_sft,
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
        if not RESUME_FROM and Path(RUN_DIR).exists() and any(Path(RUN_DIR).iterdir()):
            raise FileExistsError("Choose a fresh RUN_DIR or an explicit RESUME_FROM checkpoint.")
        if training_manifest is None:
            raise ValueError("Prepare matched train/dev data first.")
        _model, _provenance = load_local_native_model(
            get_experiment(EXPERIMENT), model_paths[0], model_paths[1]
        )
        try:
            if TRAINING_DATA_MODE == "curated_pool":
                training_result = fit_curated_sft(
                    _model,
                    train_data,
                    dev_data,
                    Path(RUN_DIR),
                    training_manifest,
                    max_steps=MAX_STEPS or None,
                    epochs=EPOCHS,
                    accumulation=ACCUMULATION,
                    resume_from=RESUME_FROM or None,
                )
            else:
                _budget = (
                    MAX_STEPS or ((len(train_data) + ACCUMULATION - 1) // ACCUMULATION) * EPOCHS
                )
                training_result = fit_repository_sft(
                    _model,
                    train_data,
                    dev_data,
                    Path(RUN_DIR),
                    training_manifest,
                    max_steps=_budget,
                    epochs=EPOCHS,
                    accumulation=ACCUMULATION,
                    resume_from=RESUME_FROM or None,
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
