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
        "F0TAbx8wfK(^G#;^xzswsUa8oT*5}~%wV3wu|)z$3}>F^lwB**)ZxKjP)h>@6aWAK2mk;8Apm}@>yW_>004U}001KZ0047xV=rQ9"
        "W^ZF*Y-eR{ZDB8DVRT_Ha%FIDb7^#Ma(OOrdF5K`liRit|Lz(84=nqIN+h1{?NdgXWD+~$X(w@fj{D(y<v=8OVvize0F+L4?|<*^"
        "0wBT1Ju+#(w4AX|1Qv_MK7QB*===V=tSETm$+8wP_hh{08zwzgB;HS%@?^m(xh|ErVTy~4<=JPZva;~fQg~&W=2^i#!K>1{eDfjn"
        "egEZ)moKtSRSM;;nOx`Ds(JsXEQ{tv@aCtiSCuGZF6GM?sVFyIT;@5CaT*!2RqPqMw;%yir7%<pQ!LJzlw4Yhrpb_0OrZp}`yITZ"
        "eAKtf(lTcG4c}~77?*_-ELIJt_q<}6czwq<RnDjQTFMOO-oMGZ@0UhFV_Z=jV$JWAWgN4jEHWU8tZF0F)6|n3*Bpt4b7V$7$Qf1#"
        ")@no!UcT_)VO5pyBkGG(*~~K4y*HVPc(KhyS!{S=TQxOmOPUtRQcki^D>P0}ahJdR_Wp<0(fijQ-(J4`<>yOp>3z}kewy8>T5x~i"
        "1->ilf!r5r%~cl9+NI4b^|To{+aBT@SRpI?$+XKg;#-#gQP#wx;VgGNf_^TLt!|_C40JAvd~<O5x7V{bNHF_=GR+PT=YR314&?<u"
        "P7jaI&Q|;=zF>#!=y;wSogA-D=JQv_rza=S_3HF=e%KkL*JoNf!U(?N1z2Hw)RU;-tE`qJkD-};JnAq#oS(cpPWU`Mou~XXVW%gj"
        "7bi#O^HX+y7M~r04CjX|Jx`w@iQw`RDf-u12yl>t_pdK!zoPo$Ie&FJUmeC5^W+d-R_Dp_>L@*iwX-uG&oAPmcz%+yj;im`YhJ$i"
        "t_jQleEBmkmLEjT!KJ9piKY)s-Y&E#Ss@iGVjgA5!jnoEza*F%G=TQDnFPRadPJGKgy<s9>xxMZ5w_W6%4yA{tldG1i5sqTG&#+A"
        "7V|=O2)R^|j73&ycHY0dpG$zoCH*l`wpZ)2aKrCGQo%pfnczqxA0*0IaZ|G!Ee7=XdIDeTJmD#=8T1ESGPMsH4NN`2aacdFPO?~C"
        "Q4{=k-JNCP&A!EEV-T=uja|ql1lh_huLOq}3jDQFm0TPg0J^N}6+qtRz@HczS*Ju3LMJo=zpM&;?9a;`7s146KJ|(cw}mjR^L&E}"
        "4}^ckW}oM?izu93e|_M-fR~Xx6A$(wKMZ~F7u~XGeH=Voa_=Lyz80bsfnOK0t`IGG5@}%x<a{H9Q41{6{sQh_?v1B$UO<oJ%F4W3"
        "!)9Rg0@GWiwgSMSnqD04bdAeL1|vREypMBk1+(*qEJ*^r@V3Jm#>{_3@7G=R{t*>FDRou=VU|R#^j2=c0f3ERx^8LqI0XS8;r<qh"
        "au+z!J%xF8+CRw;$1x(cc?&KKl-%E-o^XcSqTChkmaDm0=@GTS+z>$^jRBY`;R<NGKh>2Udu@RvLMTOl(E=`)23%V-!dPOv5CFy)"
        "z81NUYqEoR4ifvWgAzWu;}9i9O}Wy@P)Ms*Ip0Kpq>8}A@*zNT`N%S_k%@-7F;>)WhHKn($2)M^$V#}AW?XaxPDYqJSnqf7faQ4r"
        "dN=*SmT+o9>FL&B$%JyzEt;PoZmP93)r*`OVW*BflgXogv4QbfaT5*PG~;>FxHcx9kaao2*@;lXM?noDjLp*TXK${*oKH{Y_r4YK"
        "ak*@mjp%wPjMhY>ET_k4XROO;tX;t4iyUxnH0<(ll1n{77aO{P?Ib=9x3<v<Nk>&SHk~F2(hfy`Bb+$BdQBEEoeol!bu*`3>&by;"
        "C4$DoO>*GV*3@r}3brv_7Jqz~b3^YW6hID+q+w@ik}DlJxJy4Ycm-RWlWP2aCT<QAsk`TM=xW0xDQ3+kMVa)JJ#jY&mP9tID|h5G"
        "YnuaM1xMR~tp%@egScKao`#@?L^P_6LVOE#CPKo8Sc^hM(F(nTY(SHh6)yrvv&#fPVCmN?oxK9-p(tL1zvsNS9U}Np6%6<@N?=4l"
        "8tY0U6!`2BV5m$6^{_++IL0i3=MOEzT?K{*Xux4XLrAAsOAgmH%nzJ^Ms`z>T&ky&0An(oIx?;!BPImfyGe{*VpG3Awzi6iPc>KW"
        "@=T>$?SXH(vlM8W>J$2pYm*giabT3{Q}|93V(qRSBh0{mqLnV71Ns}u>@!~;9=+->)e)HpL#V2&Jt`$=Y$Knab{ajUAWH;t!lvLG"
        "@7Kl9Pnoi33YXyEma*^qzkri%GUVu0u&I_JyU7Zcd%wN^E9`EYO|2O0_TDNlSMnP$rH8wguA}mtQdz_WX-{d$Um)P3C@V4*MS<jb"
        "is?<M+;E$EUtzrf@fsCe?MiXG@K$A+gPXl!uy9vh$RR`qHVW}K76Oo7Y<V#0QFgde0_(h65x>LaC5ty4fFp6FhV<P)gOL(SHC-za"
        "K&<wvlZt~3b&hhir%j=#i=b=kD<7*As9<L^_%TZ|;nP;8y~a#(J?#zxlwJ1DRXV!`J~6B4Afv2RRVylgdlV|$aOedOoI5?;@YIV4"
        "_09=_+=rTh=M8bjZ;qW%<dlIMLVfLfUt8u<C0vLw;mAKQFfpYxG-sF_UoDR3^Xo3>iK}SZ)?03{DA(QW{6W_lrsS=QUOGU|w_$e_"
        "E@0iPKog$M=X2ext=u$6(1HKVrqZn&Y*e{;Vlo0#)wUS|$`VqV&=Q-_TRL8T;05N8Gn8v~bb1z84j}<8xcnqs^SdOw;RqL?Wm5+a"
        "P*-zrLnWjPIS1F@@<^3IGwIVvtXWaU6>$lk#~+@L-vN%v!H#5zhEY1Qb7=t(TYoOqo3btvBgbWPcC0mL@@jx46Y(hrIE5D2a-h(Y"
        "h+awP%4(g179MVHAs!(T@mxyk5b{X=sJyiYN*%$T74rBrVISv^%INslLc{NL?0DmgHS~R`hayPmA(K8n;PV`ez~^b4z{g?*1mg6;"
        "Ks0uI3Nu<0K8hRowAev<`j4SUPjDsnJsF#Nb3`!m584g1`f0N54BSK-OK!Qy(!H+{CUmMN499<b(ZBEeeus};$5@w1<sMW&zl(XL"
        "9Hsuz5E}`L?y@PIaLGC|7+LV-e?YAT5YQ3bh=jSEDZ#aTBLCdh)uF6%AYA!>pSuFlt^%yC9EVgTq)X5?@1~PD?;g`hBfz*diE6}v"
        "c$JeCb<c*7(=eT4V^s)q((qPR#FfLIDmI|Fsdt+d$<!<HU<%V#7;1T$(L)@-A>N;s-pa=W(Z_TUdD)Y*FDw6WK0iA-fjQ~t`RS>D"
        "{~WAnQ31-8TN%OD1Aq&>Yfx?k?4u_6?K=bN-z>d*;S^wN4@D-u!P-z?FLCz|Pz%E~Ym#r<Q~_5FQwC)CiTCZ&NiKfZQgNTAJbqOt"
        "cte2E&XeH^=?;&wlM#1~nVR9;=&xiHGHDYX{q>G|7Tql_?L2*z^%7MLNxu~t68%V|W6*@ppS>US_2a?in-5-%XW=vzl+AnC!6`2R"
        "hCHTYL)g-JB0YxjR^?pboiJz~s;l@GHmdNs5dk*C5JiW?*`6IqWWkVCP`W2U_p&yyPBtZ$lVPelOcC4suRT6gb!C{ziE~_00TpB<"
        "?qQ(-#fl`t6NRXk?T+TSy30~7M&vOhs=Va|IJ!L$`^x+U--&OzK7Yp&h|`shO+%h_*JEfg8qBqGM6)%?^~wC=46lJPrzTa<$;*_O"
        "8dhW1;@55Y`VkK=8MsDS#GH-}jsDe|dw3O_v|k$_;sS_B4rMz`uS-CW)k$XSSK}?(P<SR~am-$9niHz~XC0~bk9v`#a}wPx4y=^A"
        "KtN1=sCamEZU;+XG$h9x_lVu52X-!Jf3;_4uvapnB>Gt}*y*1)wI&`+-h8C+aADNkAx=6zh+hN@4_jMevMgPGfUP=ydp}}RJO|R|"
        "B!6keY6m~MPP#4MFmdY|_*2aULl9{!yy74gr|&QjO2@y@Xw2T+fl!#(4>u45mk)o$;a0mgsP#h4aeN!w6ofkXmcw{#&Pz2&|5WCQ"
        "$I_JJw<}v6VJHf!OJ80SVnc?(Az!mXw=x@K*5S8&=;M)~C$oDk-Z|wrD(n@-gE6)`ZgzIJNY?yBxof{JAp|k7s`eFrW#T(J)^l&W"
        "Us||#N(-avw#lV5mwtvwzYY7FPYk+*(a7b4Pw=s59og+`4#Z8reecU|Z}k?gn>aMal+bTI(G}TxuC@RWIX><Z-iWfUkcV~p?i6&#"
        "v=~L9&U(=n4{@>gA$Gm`bz~_~<CQq2{p}LA4vrq6cLig8?YWq3rgcFBA%N+;(|X0@0OR6Fxx4aRclEEq6$eh-8$clx^%Z~p2%yl2"
        "$s_Z+OxpMw9<`Obso8*ur6yQsNkShieEF{g1>H(p7e|P9`rHf3r=92dz6w}p46+!Sqw{T(b9rV-+a{NC_)U%-Fzo3rzr7r5)zG$e"
        "ea<zMwo7BGC-cbvhs@E*65l~(#Rb$Bm=#v_3gE(1yK<&Ka^ZJ$0rdU5x8JzC9Vi`QSeo-(h_p6G?C3ngK>A$?yH&k=z1damn0g1Q"
        "U3T6aDx>11`Rnuy6*`g)wAZKOZr1u2*PpVQZ*8<^0im|-^l=};LwGt?Xs=f~-^aFJ<2;!=QMQaYHvr1#C9KG8kgm3MTq~1fKbp@)"
        "KKFoD3!}l#ZKw4<ZuN$*-Ku+h&0V~^kvR9={{~P?0|XQR000O8001EX0cAm7jvD{~>0<x@DgXcgb8=%ZVrgb?V_|G(Wo>O?FJxhK"
        "VJ~uJaBp*IbZ>HbUt@1_aCLJoaCyaC|8v_mvj4p^{U7*v?*}T8Xgf)7dTy0_xkS11T$*HTr_+w&<4`0dv8G6sUu>(s{O@mf@r3{g"
        "%5pn5r}wTgNnimi7Q3HaEa3b8$E+w=>NV`H@&2AQUR)*X?1t4|T%=x}C9J4fI`*=<ZrP+P@|{<)ZCPhcS?#=wx0ha=Zs2297NPI^"
        "Uw`%WSJ`G;RzRq3)Zgp4Ugz1Odv2O-nB*BNx-!4iWuYD_rv7Z=YMJHTb6e#=Dr8kvR`$z^{n|2YcUhGiFDdh!B@J|~_C&eYWm`0?"
        "5(Uz@iIY68Yo-do;PN<OqQEw8u;23IM|d22AKQw31P<KcBMz8~F0y5r#QBPCHgTAgMN`E|qsX2sp1p;!jJ4;BBHq^PvT>2Y{)I_f"
        "#W)5@D){;<4}KqK=RcfXUe4Y`uRpw*MeojD&)#3mF2?-p>oR2@ZdjGa+p{8Ncj9#ducJC!6>$TD5HHys(v7xdnTt2b?<mbyz)JC^"
        "-sV{o!LSNoC`+S?C1sV?@O3oU@p8G$vjW;%HnQ(|8K)5q2xw4)UfskCXl<DlD^_i*tQh75%to^ArivaToT3`Y#7h>TWYUX45}NFq"
        "?A*eaxZ)&u0WrqtHfz>VEoMouqhc_Qx@@ZiYLvINdJXMoDM)qJ?6h}+U=!EZ5q1pgaMt1fkpz0L__~wgZ~-;gDJKEz>Kq!x<@t|x"
        "L6eA#^kP_K@U+=v&39hGV7g&0N$$K7|71zqFnzgVO}R>uaLSfm6lDcWZ6x?Wlg;wwxO=QhHE#Sp(f*F}yu6JH*4&oW_4I9=*G&7m"
        "iSHtzPNK!GVfFOn_&7S||Fg<T28)8<43$IC#Lw!(=+x|Zh~4$3v{nQBdjykSm-!6~Mpj9q_fW-nw#xR#0_A(}9CgfbiAv^6M=ctq"
        "C`^CrGbU>XG*2E%mCghye^A|!#{dh`0?_!t7daFa8ak}k@w0FK90ZEWQ_ufDAO1_V7jexZE^(uf6(~=^XcVs59Veu-7}p^X?&jpk"
        "3q*53GX%9Bu72~i3fa;tN?O+Iklkf<QwRFEv?Bf4v&xa!##zlgRJZTT<}K*`bcQA)kQPs}QCd7OaUNN+VioG$G6XB!T&GzTpsFaE"
        "dU{#4pv<YWQF%>|R?UW?nu#l%4jLI(C+%{X-330Vw5rMC+-`3P>o;z&E~w4|h62hlcy4op=vO<(Tl`@|mR$_tvD~sEaMz%geYDN>"
        ">Cur?GvNvdwF|Imo1=Rv7GI9|{2krfa`D9<4^wZl4J+Gb`r?~`LL-0;Ahz`uY&IJh7!fL)I0u#<4;P{FEB1#pZvS@SQxj`IH~GE>"
        "h{76#$4l02aqXQ>RkuRa{=v!dvlreI5C3t^i@U>5r(O=-Ad;hpD}lK4dM^i*&;bqOk1vHE<Gf{J4M-|!d0UWCUbN|oH4(c@7)x1d"
        "3#9AEc^KYSS;O`YX#oto?tj9zw#oqasCN)C{Kw_x#~GCd95f&~IxGx7oxh`j4wSv*Vz{r!YYugwy`fShN7UTFEWqr^H9Vbq&-Tn!"
        "e}0sOP!j}py{HMOE#oW)^ag@VAV&aUon9yP0?r1gdQ4SI$_T(0=xq20K;dufP%-;s;-sJyz!88f@rse9!XW~Bkqui?fxWfnz`|K-"
        "?3O;VKI7Op#27Hqj`AFHn8sNvrj1RGC~UU?H(4P0!ma95!_|C7#-53f9Fw9^uNa{A!ofg}BM}?)`AD22B!K@VGXU6I@CLw~#t{v4"
        "Sy)SIIEhQ<SNz#h!!=_CnS2EV>g#N~1!Fsvo|pU}P}=*lVA3_g9vQE=2IuY{*WY|Fe)Vre4!HirC^Z2^)45@r1%P3&oc-@ynyLne"
        "VR&)@uN4;YhK<ND*OtIZe1er#D^wom4Bldg1OEuR2A};f^sx(6eh{CyRuPUeVs#R4nXQMR55zgZD##}nS<YB$b$)!+FQ6Drc%~xH"
        "F7PM**gF}`Cnr}Sz^ZPZ91N&fxNP%$gYH-0&yOd6i6_ge`xnPx`UQ1j0G8#CI(yt&(3Z^MP7E2_VRmwg3I{t_Dw*Z|59i9GW^2>t"
        "O}5R^5iFb9L0!m6RwHtBbswVzS<y15xT)H>3Y^u&8k_szZC}?>Sw%%zRBYAeaphkj<~Vson>j5r;K#`O-f6V<aDt;}aGdkFk%DY9"
        "L;)c&Fw}9qPDs?FbQ8)G_KXzJ$a|@|MhA6K>3H-7$!&iC@W?U1I)Mcp{!(UwE%TGpD|<*z3~ojw0!h%SD?m`!*L@S*7InKt41m!S"
        "<n5zpSN0-Iz(Ov|JPk&uqfl`_j*khquz<NzoN|i7^t~olgmIb%j#llW4Op%X`ruraLJ`Oyq34of)xfA-)yRsV4{@<{wnOIxtb4Rd"
        "RFZ>tP4>VUJ?XDmLg~Ef2(^M%;c-iX>X9MG<v}{Lw<4V4Ljl>T?0=Xta<=`pSx&ySDF`9@RAgA_4HbP*m_SKHK<(Sd3z7f%m@FH1"
        "abF9ItN;x9Ncyy<!x$(OX<&v;l^l(!X(>ZOQ3|^z43DiE`cs6!e8;hplxoRW0krx-+JNIIiJqDk^uEwZfaUJH=KO7$XUWb#^?a1$"
        "Ygn+KF5i9|3AN0d#b|njXfI2mi?^4$<Lpxbq6{=T&1V!5gh@d{DN>qx0WlHqDb1255RV3pWm}Mw6X#K~X2~^pMQAT!SPg@EdPMJw"
        "jxv*SRiI3Z4T6B?<5b3q(>sFnVa;$I1eJgB@y$<@cf9Q6j12KIW7X;B`bjYV*XQEu>Btw8s^%2QXcKsYQ-{kKUWcoyY`4J)g5)aO"
        "0$A?B1u=!Z7Elv^6hm=ykHJVi(6HA9_%J>vL{y<o?3mdEPz?0hW|i0kK~1z~6<h4Q6?kk7?wu?v#=LUrCF{6aF<LrAOpBuyDa_}Z"
        "d3CvLZsUrAQQNA#$<Q=e6?6<BhUtSi`l@Gsyp3Tt<?c;apSJ@(1NHP-c-(z*(BxDmSj;)-ilrIyf6>zR1k!G6;xRX|^oHvIMCF@$"
        "&&Xl8zjNpop!64YI(a@x@)-K!_f%Kl7YtA+umZV_&|TvWf_Bh4Wd);vzo8SGve63IT9+gS+k4h;??E|fO5_&mlI;%MtuoD)@Gr3A"
        "2%1)mWuC#CK@Mhha)_8<u}(~gH2gowCrw|$iljx0h4E22AE0#^*zFYPL#L5%+jar6<)!5&&WgP<)D}_*4b7^pvreZ_kUmu^1t(`L"
        "?cbOybU7y9_urNn|AlVj{uR{8Fj6|vHapHQF69b<FG=2}Z0zwp9Y8EzhMRN3CE$A^#1^r&gihgBZ13UwzGsVmX4C~|zcoVeOA@UC"
        "WYsrCqMMQEW|Zub+;IIAk|O6+XR}bmDJfiAD@z1p^h~P*z(&o4-AO)Cd=k1UV|$}O;*$q9127g=?5wrAiK}bwrb{YLua7)>Rn*}?"
        "`mj>BRmK`b8;(ewzbaa9#R^t|Qz2wmisevo5H`S<2iDR$Y&&RH>{HBe&F$kdE7Ca6!9f1&bND&*g1?@M0{;f592*voqf;=IPlL}F"
        "Pe))f|JT>n1(5l{WHh1$*$#o%ll^^4VCiQVuEMZ^6#TFNjZKaVXrjOjl&A;^0_!6+CUBs2Sa111sui=dUE_$_V18Xboz-DOGayEm"
        "G=!Fzs~1;>*5UP(#6MP+rKY0rj9jt|?NpJlpo_|pFQjRz*9feXDq7%9R1DHulVA%|DR!|z9vA{&=taIm$gl!u$rJnDotBbexO<9>"
        "Vck=D7fPNWWqd&zNOUZuT3<K?*=;SBhjkhRC=HKVL5H+y5MQSDwDR}boC0T|6i4y!MHB$GTU&jDn2_uVP_bk{%_+qn6<H`a(t=!Q"
        "llNA%0%Agl?NFc?m5_0>3f~pNtv-kEgU&5MM&Bn4+sM7b5$*x_q0do+{{ATj82Xg*ct?ch?r@iB9tSpFa7xX^A{ST)>Sa^A;%j1h"
        "?5HLm*g$!LAV<Dj!dQ1sJ`1npojSshvYw;xaPW^hSLlwG{qy%IcDqd2+%*T4+oT5-tLlTuO%SXfQjP=gLrESw8bS8K;s{IuC>>H}"
        "AFVI_*A8hwvxNdb5ZpD^a7_WQfXY8^$I-bW2bvrpp%e!Hqo*NcFhX_sLGUM|rXns$DOys9hoIj4FtqGe0A1g%jD90PlU7Qs<RNF7"
        "!0%Yvw6Orfs)sfRa>Nn)gN>lg2$Kn$#tg?wTP1@+CT)i<@F_NWDr}Yu;fW?OYBHc-?<Eld8sv<ec7yggY@qN8Vg!ez<FA_oRa2zF"
        "7(8<<6RYPMeKSra8Dv0|<ROfbtmQNbG8!lDea^c`FafQ_oO9%AY_N}iQ7^8vPU67vvxE~5_J>E21Ku|=#H(&lB*J&iJ~A~E37;DL"
        "`jJ)OT6>rtx8tBFPMQ{A_s)Y#x8#YVQ_E3bb%cAzn78;HECR2rVsOPc8q&PPW!Dmc93OF71pYh-u8c)f>}+oY(jE|Qf@QF>Bf)(1"
        "%Olm0ToTlmP7U7Q0%$C6<?ImG`77>z8r|d{YJk4;tI-!hFusYi9Fs5&MENj=)bh1z8Fobnnd4vY`)Nn{QEX5Ok^0VLqNNp_Jq(5G"
        "8J1l*_ah~HQdaFdmadf9feC)oR>X=et0raJBqg}<Nm*^%+M@vJw#<sA{?1EF#;3W)L`{s}upDz)o8bgYp03BYuz^nvjIlPN20Zpq"
        "|BAlfb4fx@xlC!aGC9as6Sn9vxoMQmkS<V;czdBp?zI~b1w)Ze8u+izXRj`2-sP*my_<PwZ@u>)F1^`LXBU?jYE*$rG4?KJKV6Re"
        "J@l;b-5%;<I_jWYE}<pw?EU5J@3V97<N4VSug-t={yF=3yp4C18q1sQDNO2s>f6+jd>p8m(Q4onr!tpO=f%bB-R$+HM)K|XhaVIh"
        "{86}Mpv%gFS+Nj75%DkoQ*1sfLJu0yPMAgNGS3m0O=GMX#*~JR1WVqc)k1G*XJF{hFOkU~I;uaudiM#;fACif*LSl#9yunMZ<27w"
        "S*``lm8Nx|%RMR@KBc%i`22CzwNFq4>qSLbkN7VI#rtsn2IS*!KZ{bjJXpiuWegl`*$X_15F$M^Mnu>cMIGNTv6`)Z_mYu^II6m+"
        "I6CG$HM)?tn{C&HLz$f9WzBkARyF9W7S33-S)S^7?_AvpeskR`bJ=0~Xj4j_6QEcI3SI=7N4#vxYleuXiW2lQ<N<=FO_6l5%vRzR"
        "@L=0Yc&3+?8Ro(aCYU*dspBf=8<aB-LF5^Xe)^4k-&qoT{o?p9e;!M=kZl(qo*lpVR({CpYnckL$`(*Zd>GDb-&W-cjC4K4&1THv"
        "*J9FOuzv%5;mIt62}OHgrE{KsCr}8xOY&Ck7R4ec4W)!s^uy74+~5w1;5ugg@h=#>+OQ_ZvkkJa*xr#6;XIrIlOT)Fegcqx_QUM`"
        "Wpwdz_WEL|79F1ub8DPOp?tp^1X&r_YBl>qbQ;iQ1^SGd#Azo)Wc~g@3^^cFpFC(LoRoXNzC|5=$0{FD`7Ksn0329HH_!Yp4`7dq"
        "NzcpWe_;AfRs^~vjxF&Un_@RMg>7t$7^c;n?Cn!l+{Ag7ighTe$_|0o)vY=(Nxc&_fw?FZXj<Wx5@w-y-{(G#qH6rBxljDUD0}p4"
        "8{@_$AYfnkhl{+#jb^d`AxS|U{=S>43_Spu`Ion=x>qHs%DP7OFc_mg$5dQZhVjQWclI^C!TTGDkFs}MjYxRJpSs!O&#v<X-T8nS"
        "eN={Z*&_Tzw*b>CDZSY|M1#|!3@a%v**ZPw9}dQ3fAa8%4?kdssgBqKaMUT5kh7TYSH?MJuk0prhd$M10b|E^p%^&+O-YyLF);m*"
        "4-6(=@f@uwb>0%3F2L{0BWMxPW%ZGgmOA9{QIGym4k`|W50`(CQbA~W>OAelxrgn^Gl;O9O+9Qp$&I+Isxhx?iKrR`1?6!GP1uSJ"
        "AeM;iSL6Q26ZKI@AWtt+#Ptj#j@!DnddWws8SzJi)F|V0Kv@RbAb0@UG@{b<m$-EuDe}hS!;N%Z-rtH)V4Ws=ka9QqF0Lzj&PEOS"
        "APrmh(}m*%Gn{zrORRfkMD@5nDTpB=j`U;+Y^O%@j&L|nP4uIEA`ld*_>Z2~tuT_AXVa}M&Dp@Jnl{$I=)oUfH$N(_C9IKF%(P2T"
        "&@xn7QNv_GNz;UB1r1|{p)SL?=><(QM^Q23jb2!t8IA;6!9*#|8p!myo9B@g9jJ{gDcw3)jCHwoT4^(rN{J+w^oL2A-oxBng6@TB"
        "FDEVdYILT8S<N;meO74G-|m_<u}b2>kiU^k3`SPO(5FKqSP}|AJz36t->)UvVb<y1>NUP}>0$U!PVp32PkW**8uygS=k;nIPYLex"
        "%IWN&usNeHj$$a0JVx`%g9KA=aw46T&drpO9n4&zhC}j9b$DPB;eTlTCKWUH*<B3y5b&a%<jA21p?QkDVMGzkt#3P?DzihBo~x2q"
        "rswW+6wG%=dD&jd1{T21N>ZYVwcjFjsf?%dQoa0VrAX(z0D!l+D2ptSzztyI@s%sNQ!+<3)MJigm$uX9vRi5t>XUCLUdx6m_6tP{"
        "Pw}9ts~#4KYy2w|=N(S{k=1)V2OwNKe)oYU057NBvDRT(F1Ys%H>GYjOjN;*f{p;VC5jn5ol`!lDD0oS3le)ez~rGnSRt@T8Tfb%"
        ";XU2Fz#T;jO(%JrZ7@dw)^dt>6_ndH7g3-X@FDtGeaWw4iVL#>x{6c-`zX6+z^s%e!He&w`pE3f_Fhh}NJvrRMPBL#?19{rht`=~"
        "$zZ^>t%ppd6Y#|E0EW%iVOD^4DPXKw>dWwr9$4uoW7B+02hZ^b0wW-2ZSj=rW%wxeSh_m>6m%++$Hl6}BXY{u6-Ho;=D=`n<*)Bh"
        "Vy3sK)^DckB#@V$9tezjug(zqN7(Ab=O+fxUpuJv&a)&QXzr!sY?G#SR0pn5WK1Y?P{Xo{n6+QqvWa_bc8BF1Rwqz>828T=@;BW+"
        "S+{4U4M@>{%pr@k#e_y!oDto+63N{>VL=8Ah-}LG-nT0-R$0~}<vN*39$TpH9H83<%xd-~3_t0<fB&)1JKIL#7M%ip-F8=>fA_`~"
        "q}Tz2;Z?-+Ugz{(Nh&q4dx<~$poY=}v>xzLZR$>a@74P^o;WFFb>9{Yy#9kCIoTb#GzVTj;ut*jq5s;tu0;Py=du6A2<J@2lcSu_"
        "SX1$@qXeGKlDM5<V?7;mFkEbQo`<=~>I~PXL)r#6hWk_u=HNWcl`GV3v^w%r9y+zP?=1N6jwiwuv`$oWG<=?W!Il`O?9%>Z&Yzj^"
        "H@H(Pe<W`Qk)e`q8kiZ6`;mNCU+voWSF|%duA)n5D3Mnm6&Yu7x)*W1)>HG-(D6^SWJnm18;^wX^Qri9-~9~x>=k9FlstUqx)4F0"
        "q6f4pj{br}hUb!tRIx}K$UrZKZ}7k$AT12U)p*QEQJN!l8w|^zyjJ02S2Xb*zr8>kAl^NdwB*y#^MltIxVa>Q)N;nTBU)3Ndu}G!"
        "#u*)`8I1Bf)-1m)$$skyG(Ae&$)MC=9kE%oPjP8<MCqaM2>xutOfiC-LQ;s3jhKPVLDn@Crrv;|kLoQ;GO#W}vr-Vbk32f^FU{!A"
        "K^n>oVBi{9=H;!u2xO`ZCv0e9e12*&2cEw<dFH0@dt{<70xy6=v@6#SVe;uQJG^&#papr0gTqvk>Y7@!<1m%WhE-GFl^aQMcinxI"
        "bR1<Z$9t+_W0xK0dR6VtPt9B3ESa*SKltlT`3f93m4CPjJ5u?i1)kb3T!WMLcORW0>+f2SF_<Ez(7=?=#%a5TnDyCh7>ZL|NitdN"
        "CNIKc7yCueuGwzd3kb^}NiQz96&S<mK!UO_7L<RA5GmkC?K8R9EAS8N(YRZ=uBjn1SSb>cE#e78M+!UONf;fGgGHx43tdgGC*H{r"
        "bzV%Tt5qDTq*j*ME&1BjH8-WFW<9NGcX{_fzPJR9r^N8sJx?m&BisjK>;d=u6f2fi+ChL`Fph5CxE+qRX@2kT2(95yi)1;Jy$W8r"
        "`MEE*zLzB8+9K)I;(eB<konV)98m=dFKO|zE}C<Aj=7#c?7U}6Lqz6>f3ltHb?+iM>-!=1PwlyPN?vnw8CUk_L4?L*nj*r*eTW0%"
        "qrBE|98_bQNKye;!Ed+x{satp!IY@5rXEi-X3CQx*Hh8rmjUuB;H%yYtspI+3w=^gU1<@^r~E7{)NcmUEm^4BO|Z2u!y-qA9F_jQ"
        "^e9~A<s$H(U@#0;TbS{XY42W(m-a50_r!bl<jHgG-a6Ymk&{+8B_7w0Gy=}mZl5rGT)fq8&Dd&upgF;Al~G2`lka7`qKqFzVmD)c"
        "k1U<ycit!4b(>e^`6-9*&T6TUk;=k6Rl7w0^^II-md&Zk&T(#$IyM3i(Wu>O!Ghg`T}<v3MBR-QYOT{Q1dl<f_$!4`qqvZBMw2I)"
        "i9)0UDBTM4{lnTZ8p7g^4}25~F+Jpn*{EorE5;|I4_OZIUxp857^5WtN72K7ty?=N|1H|)v1@LD$H^MGVZ$2eF9H?RO|Gx7a@vj9"
        "0D|u??C;!{0f}z&JfcRR6hTnx5dy{MBk$;_U$bsu@&SM|x`Z$*A~ln!H><gCU|-y969|%rV(HX2Z(~R_EEhn6{7xbZ7K4y=5k8<B"
        "u^1ElgkL}l_vXY*4v=0<M<?5CTCvN9*Xsy<PTOhSO(+vsn>ohpd7?|gS^7KjG3>jh9!!g%tHk(N5Q%Cm?qgTr?o02a<FL}4=6<|9"
        "BGldS&F7nh`(`YYaD}GTPdWEv(neyip0I0V^8T@;V43^NIXAVlIdyVIR_-6n71mRN56u;RrkWVe2$pvZbJvqj19)4|#2zj~kKg^H"
        "-0dULrz3d^&$?QlxQl_cb-P-D3R%V=TkBR`PrqK`le&a+I0Ja#D)fsFpUz*;qVw60XBTH5-e3GjGHyS0cbMr}mArwlJLI!ul?d<K"
        "-J0{V^4L182)b-zYTUE}eVR}QCnWBZlau4?5R>Y2?G(UJP(bq?C^~@rI^uLtvW_R^lKwcs(Cn|RI$5Bd(84RV&FRp3r$%Q6E#***"
        "TuRh%WijoloLm{-L9(0O7FLRgd+3Ua)*7zDsg|9JKz6;<i}BS&|24BUh4R>cLZDl%8cFNER;NxF)(Zo=0e^<7!Jjp<zKg22rYng7"
        "t!?Rx!4x`Ee++?>!=l5sW*IgSW;(2vUrSw=dPx4!&F23DP)h>@6aWAK2mk;8Aprjuz0CIr005>O001EX0047xV=rQ9W^ZF*Y-eR{"
        "ZDB8DVRT_Hb9ZHPZE199E^v9xSzB-2HWGd};QzqYzDVnH<hFZ3;o@v}7fHGW7TqFo+lRzGpe4$OjYO&>75g;Ef8UuQB~h0gCtVb%"
        "9}<g~;czbBd_yNm^5m1huv!<qW`$PH@WQb-yp|<*QmZeR(vETV;)kbO+emj{rM_3S=A|%98|H*{?7)o@*5>DD=V#->mb>D>q-D3="
        "-BzlyE2O&1+4TX2N>R&OVYn0Zkr|L-WZ^^!hnWV>{sn{|xGHO5&(FTQezyIbHM*3QEJ!KKbj8Gjv<@V)O2V=YgXQdvFtA9#k$O;|"
        "ltvVO*FCEBm2{wJk|e0DY+7v`tf?wcX!t&Gdx*b(vRVxvMhw5&?zT0$fV&SvPBblQbbeMD-7v5sFKTWrDAga<4}~S5CLY#Le+!?e"
        "8qM9x3O#ctnuh1Sdj?6*Ma!jm`oNpE790H3Ryu8t&*b#`h2pI}Xoqso&q`4-8%Fkc)wKV*TG^Aa?cW(V=<588!B5MN=zX?hFViYV"
        "-8ZZv)r5kz6~sivq+%8n6y+c}4_<mLj=P#Sw<TxQ74rhJ8!uW`tQIPL+me)a!}cgB9QZC{f7sD{7|4P`_+z-Vf_=~HPCPY6n>2aS"
        "wKYbI7;Z{-wt15IZRiEK1bQ|vyQa0h+zk`Tc1G;EEu`E%<Fy4_S}-+)t2TC*CYxl#lB*=ka-j-ciZtn5wf(#oZG`KL>Vqh^2maY#"
        "|CZu`S$+@?rM!cHPc!d~e+=P}g2#UmYIkiq0h2U_dG`Xd>A5JhDZ`bevW}}l>}7d{t{HD+h1(nPMp}sXXbLj$?g$grVr+;5uVplU"
        "Dkg#&)^=hlaZfoH#y;SB-}L!ju3l95(7u{R`fqaGN<N74#Qbo%<&+`i-Zwj9DGfr*C4QyRohkwN+PzlwaSu4V*A+mV<n<n5)h1)Q"
        "?(d}A^R}&zsmC*L#uY)uhK(lQus>};8$dw|HoMijhSNd-7Y9+io(}dIRA@jKMJb)YIT+mA0{;b1h@5v$7pAUu74E@d(t}oQ?++4G"
        "ZFwPfO5Z~nNXA_56&ces9F&6qp~@8Yhc$iXz%H-nsXeq}%r)E94pbkZI5(Y27XsJ?+-{rPY#@8u&h5x*^ZW%f-Iu$;!tlvkhghIH"
        "hnp~VH_XXKz`*X$pZHU<=*p-~?g0QLpTwwdsXqCd<qI3DIKU(%bpq-JVb(#ZHF+swO~LpObv4)>P-qO3p?vP`y`U-#S%CSJ`v0zG"
        "ctQbV0Vy5D4?+DSDG7!1xpx|-M5wW#Iq!>a8OQ`B+<+orwS!Q8%`s#Q@dcFmL_>8%o*`8Ql^+t<GAe&P(6O(dK<y9@fJ1@_$fypX"
        "n8*i=!*Ak26diy)+9>xzzXKF+TdG1(Min>5OJ0_jz&wN#mqsYw2(LaIj&=tRfsP;*m<O}q2jzgA3?|Ko`w#9$mAzr}IU1o-(e!{w"
        "vvGpZ9O+M+K2p&90jxiO%mF1%x?pJ(j&V*R88-q(e(oTY%RGrAl+GQ>@+}D6V~T(rsIPN%PpZ^c0~QNuIv|9y1mmHj!Jv_PP|Y2("
        "zbV^{)tl0inX=hJ!#P}^()M4Ic%<sg8fhECfk=g^*F36&hq!ST`D2<OS!CNSASFvq`|RmMD~Q%|WSy*2K0i>vaN=+UB1iyaZ#^lx"
        "8d=pys^;FKJ@0~`p5>yAa!TsdB<Yt;mVUYWF;Y59Hj4`eWeN!WJ3*Bo;5Z%~B%=awpc5wjMl0e({>0b_2SJ?y=|H9h9ip@OW%}~R"
        "SN7&2W4FVvuT(M!%SKVb7eONE;eUJi=?%s~`1tH5I(*#=|Ap!bw*R@)=!(8=*rM%bT?Ho12pSZn^>OeN{-9^rZq#I)nUe3+6XPVA"
        "-iNCr%osO`Cpf?pUKm0fV^yd9S&l;!^RC`-I!z$;Z&^YLvPL*gM3=ZP-49I~0i#b*pO8R-u-Gx!(t?-sWGZ-i6|#iK?0F2+U0@bw"
        "LO?z0BVP(>*Aze?(8f-0NM7<Cki!BIT0?=zXNInlS8BW6LcG02q3>R)i;D{gx&A8|RQF-`sDl5TmySzcQ-%r>z!p*+XbAMWlre)P"
        "mtinHu$D^1L6Ii9E-^X<!&<Mz{rC||mnUfLk$(B=>+4YDh!4b~F|<&8bX$)OIB3&o^`n92Q^|=3_tA+H0->jI6y6iaT}H|Zo;Ixu"
        "79WIDPv1A)`L)2#7|igb=WuBYl}A48lDL)Ll8h}R9=Eia_C#k76cCb~@^M%c-RY0)2Ltq%A$<=-i1O?UrjQ}Adz!VnGr(YowE}C5"
        "*$VL&029-bi{4+S(+9%jP9WzUwY;P)Ys(NrAeZvyHLOnkhYduQ5^Jwt_dVIlZo0!BU7k*}{3RZEgUvE<E=bPZz)T@nF|>9Q@LOwO"
        "mSt$@UV?&Tw7GV_MpU%jH+p@wEoCvkS+v&k)TkVOn1d)L)46xDTtn=qCk=-22y9UB=4y%Rk>=j|*!Iw3=s0+p&O8_rYt3!Y_mrXf"
        "02XFFDP0^eTeqUqGD-gv*OCzz!zH%>hYbJ~;)H+L3;98(P9S4+K#v;LluCFok5PSnHv}*4UMe4+{qjJ}_V~TB41ZRw7{TakAy7r1"
        "S=nY?$kM(aEdjbB+`p@m{{G2(mO334-kw-EGuR^ZeJ<gfMUwxdrAqzq1O&hLJst9hJE7=Nx9l^AN*X=VVmpURpk{ZA?(n1FZzZvQ"
        "Ym^#I@sXG`Zp?lM0D=uxj2FnG7csfAkK$pG=>AO(lBgVdCTQ0UP#<hfed9yYM>qePitvBbfZwal*JAX8%?^F+Wymg{oWncfvYqwj"
        "->tmhrg5fEn@SL_BdAEA(~)mPxg}vSh`6qWPG`08nfyIqG8Fx7APT)4^0xspn)SdpeM*1dn^bg*$qR8huLJ(EWqdZDL1BV_wYmi-"
        "A1;T#U(Ml?pn9_^czEPmKNcTk7E=+VL=1ehaLfE*X}Mx}ta;zm?3E;1Hhpd(f5a-x{(`Z16`8%BPX4-aO3Zlo4-o;O@IO#X0|XQR"
        "000O8001EX&twv_wFdwI)fE5$B>(^bb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~!VYh`X}dSP^FZ*DGdd8JrObKJNQzN<?A0SjL^8;X*="
        "o6TnIan(NTln-&XY*j9@iiIIT4J;C10ML@+=)b2M4-({zq)J5>i9~m!A76jn6s7d{vme&vE7zhCye6%xSWB$B;ZppYYr>4-*5pJr"
        "ji>}1y}uxqcYVt&&q>d0wL4L=y(gCGYmh7`rN>9dN22SMwq(c5t`*zy`HNBV_^8pUBmNB>7MdG1=!%>0jx?g>CWyfmXdxc`0gq_k"
        "omtJKX_W4`9(8{ith$KqkCI!_U|`q0>sV1K*@)|?@Y{}8H@y<le&kI~-rn_)i;heCspr)r>Tq#o$16Ybo(cW-jzL-=De?At_8wh1"
        "KB{>`&|zjvE-O`YU7|UDXEW~Q)^l?!a`J{4tJlXz1b#G^paXT<5c*kumbCZ+o@f3Le;WGY#S8LIprLP=#{v`h@LBo?cbT*gG|^Dn"
        "C7cLJ41~k$=t^NE>&;Ht!WASZH~fCnvTj>5r?DpE1uJiF6}_~Bmcf_z#ta<pk46PPKcMs9Vl)qQ%1*0TT}C*TBM`G{EYo^fvQen4"
        "7>KPMNLC?ZfXCnPEzwF@g7GsU4bklO^jfLwmY1a~>&cHNMNz!+op4XK_ZIND<&Yijd=m1yKp)ZONB&h9p(GI?q6S%m3&c6d>k~+$"
        "YZ`}r%$kIIg>1OSx;Z%2Li5TheNQCoc;StH;PzJO8`5y-DY%U=u&a5eEKm9oF+&*mBsq9WxMbUw*TopC|D$74G~8H#33gn80hkme"
        "SkH{kVC_4dt{kGUrWu_Q8x0m{4rDZkDtXu<jKSRv9c***Go2%WLpN2{;We7LoGMv#5KUW(`pQd-W>mPOIwYB1k?%GnQHlraDIU#)"
        ";pCsJ9r#<VmCoo$0YoRo!cz1=H_!qIcMGs!LoLw3F~S(ZRHV%kVdRuh01$=W2Z8|yfgUTK(NmfeT9(lwb{ZBu2iw<J9dKca*VTkx"
        "72r*0vQ-it76Y7^96EJoCIQHsiwxwTy-Kt+GcctHiQ$8??+~;;Qw{w)bR^;pD83bvFUcQx#8F6$B3u}ly)ZlW^3OkI?rw()*E!j0"
        "RqXhkk270g1^wR#y-Et#B$dM}NGD8(Q)cPL<mG10`31IE2~`G?hH&Zx_G1QfKj&PgOo{VrbD;+)jMam&rSG~Ddu{WMwT2&Jx6y%o"
        "EaJps@eoWQnU{h5ve*@es6O|H25hYtV<=z5oDy9)`#z7Ye>aX6CvVn(Gh~7v&;I&TB2@DygxuIUJjf<qjpyXkyNei;c|}^$2^fx="
        "e}SYB1vJ&P>K5!93-|z-5d&$L7MBJ+$couHx6byD_rDuYcaFR}fPetPBew!t0^3)@i?4uLA@9josOffS8FUC-a{lt?K#Lb{H92WC"
        "=j578u0aR*0IKXp!TYmwPtuIkL)W2K@E$wD?=)lwGiu4Oq2<oV>jN1luQ%lUhx4<)yd>E2iQGKtlic%M7Sq}RKw&Pq3;=(e7lpSl"
        "@>cLHj$0o?={B|K&7rdbmJHF)Xk*_KEOe+%%;CQVXey`?7?!XX+PDe&-Ed-4A9x=BD3cNO*vQ9DC`QEv_=DZBJ%N>{*Qz*=!l|eY"
        "sAP+`iBBIOnlYcwksy@hC%nlI=|36MsCYQ0eMjwp<PqKl^(;h~6IRzH+J*Hz0L!u2YEIaR#mh4{Q=`4DGF}hPp5tfwOPH5iX3|kn"
        "i#kr%9=1C2>pwmf>BQ)PlzUeu`qA^J`)Xihkp9D?77ftmj^Xxy2Y>=}!-9>wA{pFx!fp>9$Ep{5Yh~z84xGLIzb@zg?!xV<;kszO"
        "<Hj%;W?=C08`Y}ibWP}BxW;)OR&03XuYCy14P*w4;~e*hE?D2U_hkjQmhp=!nN{(U2_$|L`EpM*R{C4Q{PO!?iPUg|OWFSPLK1B`"
        "?3DkjBjitMw6GcoV;RIb3;7VUJY40IwM8(~!QrTuQU~M9X8{KOi*W+4F>MXfW#&upG{q&~x1?4)*%>v?ncZxJc!ML*{o|}>V~#`5"
        "dWnNA>q3b=`+{7+O%(%D74CyPx&*a?_Hum7Z{51a8VenPj#aJlodO6i$lpdcqe){`51Q5qPdm&jh|@*s<{nsfZxg$y*P~(gtpnpV"
        "48vw<+oysZOQ6GSc|L9^UbK3-Ca@)BQ%+Zow5I1L%lPXbZ$liFR<OqUp7zS8VM2}Jj60=Atq*DRW8no^U(JA)D8YE0o-R+$ox?FH"
        "&?GPqYzP~Ep-@I&V<G<YhCYi*fMBmAymuQw0~-o#$Dx=LP*M!|1f+|>6h;^U(YZ%Pz|-yF95Dm3$BggcBFtGba8~5ap*0Eeg@Q5D"
        "QEIW7KEE1mhZ^h2?<&q~UcZ96;QwY?Q3*TlnEne8wrckeqk8l<tn-JL-Gm3nK_ij(Dh?jch-nNqPX|xTcS?aHMDmnnEvuPjn{V?0"
        "OrzoOdA{CTHu8l+%-*Q5+_;ts3r+=%CF?pdt5j<qJ@iTD^<x5IgqFR6s@%H?(#G#E|6eEgaKw4Bqx=4)aG>iwa6OiRnZgLNrF5v#"
        "OgL%xy-0oaX5v#t_r&*64?6oV52@{ivHvnrdzhx+e*sWS0|XQR000O8001EXr3N>f=>Px#+W`OoCjbBdb8=%ZVrgb?V_|G(Wo>O?"
        "FJ*RNY;|FDX>V>XUtei%X>?y-E^v8`kIPB}F%*XP1M?1ttQ4d#ATF#PrO=Ktbs>m^OnRmUbK^`ZeS8x~3)T^uRmgwI_X{Cf>WRr~"
        "f<Xr+V2Xj0@muHv!`$>iD|=7QsY~dx@69QG2ilX~B@;ZD0enbcT(soKC)&(}5Oq~oJqHKfSg<VxGx|_=r%LSWDs|XUN#+mXtgav*"
        "e161svBKs_t{!f2xxN_RTfD;-n;Vq#>oqPH4gM&RjZeACMa5<aUMAI9`X%N~%J4}Z+DBqjh8#TSWsH={S}EZQUdEL8Ln^-e#02X>"
        "o2EUI|KXi>9TOMVYh8T;P)h>@6aWAK2mk;8Apl&&;2mKH008tC001Ze0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZBnaVQgPx"
        "Z)9aIaCy~O`)}hm68=3v{)d5zB6i`&X5kh+=Umhe?bd)_u)W&@4uwEVv`s`3U6OL*=JLPakdpOucC(8f+Dj0`7KcO5H#6Ty%Cf8_"
        "J#e8wMu&sXs0RbK<u38x(;y%qT`s5u;`4APnMXm=0}(_~{{)gp!o>-KZmFQ5OJ(c&>iTNEXA%PLMLq?O<K&PlppVpz6op@+fQ0su"
        "_zjp27K;Yx&kH}24d4Pj9k^tQi<ACHm;goOy>`2ZHa?SkT<yFgLBZvrqK^uA=%4P)H^F@$ZCoDUE}7(^L`T0z42=a9Q5dptcYU=X"
        "lKLz}FAq%ZLG3Xt65wtLE=N@Z_C(+)?g<|wDB-RFB=kV>dm6SC=RRyAZ$}kw%(5&U^j7e|%y)evC3<FoK{b7pn5feM-JM702gP_u"
        "{M5rH#(t-<Z(%@NmfG6?TG-3+)Sj)KS$}G;^qlLfpK}YCe=tabTN#|X*H?f)mQHh$r3Aq{nDLN8Kna2rh2%(>Pd1n{%!AI@kqWle"
        "^<Xwel5Dq<c(qf^G4-7gk?Vp~B7V5n38t7-=YYGp?WQsQd+T$eeuJJ_H}SAJ9A0DRvm<qJwGHln?;L;lw6%PoPHgEjMPF-sj`!2{"
        "uHaJYt>_Rx<Jth`L?Um8IAvD)_q{!x_B+mWKC&myu)iGPsI^z>AiM20VXYnKJD;{ZdTg7jXnWk1ZJoh(jK52S=X5YBt6%MbJsbYw"
        "Ozi&1UJm9ZXAch#EezL0i?tQBan5i#U#*6?w*9d`ahCRCo}aP9?ua(VmIUMv9yX=RbP2UxSI2KE;TxD^`(@>l>rWQCe&kW_4(lr&"
        "KQ|wQQLv#Rw$ymEXYKy=)zW@vFEI?;8T427WIVH1UGSKzG<1zpb!#Y}k_tC`)$BCww}a+*ZFNhgwb<;EX`@nKy-ug~rjg5Fuac(N"
        "UA<9Wz4Q9jJ1?#N{oB{C{v4OlH!vh24;hjNGTn|LSV+*wV~DsRxtL5Mq;4$4n9Y#(hLbUJSdLoE5-WuDGTnG|3(l5{fGEc#x`t&C"
        "ReP)s&7U+UB=jSwibMz&X4nF@1v3dyCJY5pUsBjwPqC}k?e^KqRyzn5Dkmd!M`@Bu@I-_hL^tx4%oj<#hsH&g()FEJI8y}pXC>y}"
        "CI6X^qNFrjN1V9PoZo|Y1h<L?COp0Oj9wRdpE)Ts5T}}Yrd+a-6wLyv1al>BSsr|H0)j}i`>ivvuBa-k-&+fW3t>kY)q<4u4vz0B"
        "s^p%8n#pA(j2;_AT8T5<1L{Zm9Pa5UZn~7J8-|EujVl*pD~&Mz@W4E*rXKv`$N0O+PtEFC@LfDsPbiygsD{iMPbME$>t%mE{%8Yo"
        "&C*^$Z8l%S)b6i7EbUR<ltez>xFW38hs9ztwnxb<E5G2@0(dfOJa^$K%ki^SFYF5-)4hY>io4v$(+Ohku`LeVA$C}Yrl@kK6{#QK"
        "c|&IqiHi%SPOXybO33G(J>5+8er~$|eEZ`w{0E+H(o+xAwA%fATi1hc;z4~5%0IvTUO%4<2XOoWYTwpN%Qo}N#v{5ibnp#Bu1LFZ"
        "fkj)Wg@LTqyBQV59<%Spm-S?N<-GS|w`>2clEl}~p;#497ba(tUq6R>$*f$4_2-=TH1JE1TnY@glB0pkq2;)wg-5&EO&aD+{ZfdP"
        "@G1^;sAg^)<`>N^$B%x_2)wci<MVq5Cg^F1o(9^JE~%8GuyHEFV*01Zz41x>|A>Ok#i4R>A^EF|VUFaY;8P=s9wWPuOU2xbhg0g*"
        "oY^fMEMtbRoF5If;?mqdYR<*=<lTj{qgKD%>PYM^>H<;jzF{bcq=7_u*pi0o>EEA>p@xUZ{&)#(7|y4QiM<}rXD}M~-_Pc&^>|qS"
        "o3VAFCo8;-N9v}@E_M+n9LYGx+u7hpDq4lXq0Y2!I>Y())X?~)?)~e@yjas;#^u%GPv>7UFVz2EK<LF#zn0ajL4U2M0v=yZ@LB!k"
        "8U4~Nu+(N-t5}~GNf2FCFNZIy1AWw;7vSNxpbkmA&gSHpeB~vgEEd20DfF*zHCy(m-&-Ky%_B+{2~}G?@Me1RJTt3`yi-c{ruQX8"
        "*mK&4n_j2JBdNQva!G2zM}1+5rxa_e@zTqM_>5f?f?8Pynnnfy0d)~ivZM8c_xpw0Lwx_wE)jTxxGP*hP=_=<%4mZLRgp57i1R%`"
        "=Sr3ZB?|9D{t!Be>mF*Rs+E(7*qi4SM3O2)0V(>)v6LVT(K5i<LU4h_m;lcsWpvuSjbo2ZGd0c;)}k~zTr2vmr-l7L7Pz<xZ{KuE"
        "eU!U{yqkYmg68KYb|0dl6>z+6j_ml0*2ER`XCqkuI>!M>sk|bAQ=NUxRVIM(iNd_b^b!da@k`sv`&CnJ0TK7IB!FKnbVW-0^HeWG"
        "(-dXO<`;V{S3&2ybFV@!hE+314S0I4e^wtqD<jEzua}UE<o|+~PfN(>J-s9$FUUxY`F{aWO9KQH000080000X05_!f&KeW|0M|zV"
        "03-ka0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFK2UYIBPC&d9^%ibK5wQ-&2+Uz~g;5Dxnq0Z)>92n#ws_xt%<2oY}f6N5!H@"
        "NaC3yHGKFHm;d{9HvpnRQnupUdTTc#(P%XK-3@SE*Zq=YnTVagUwl0MhqEcfCM?1#$ulR+Vy6(P2urb7R(Gjz$|x_AZ0S3tgc8o8"
        "$kz_eNebuU{nw%Ex-Vb6e37g-c~LoGu>`nMG%r?Rxk{6H^ZdVMo;42zz!APEPg4;g4W*;T-sN>xi9!Kv!V1yo!7uQLtNwBR&kq+L"
        "K7Vqij(ad3jiZzJXmqlFI6f8or}MMX=rlY&J~*Do;&eY8jgC*x_9Gzo;`5(>ygLuxfB60U0;b--eBr>KTNPoFxf928$0v(}!;?5V"
        "8XumVL`S3eP%P%B2WLlT@q7+1j`z<_!g+XldMu8P4iAOc4?__jjP?&t#|MYUN3QRj9G)Bue2rO@6(K%2IT*zU<Nd|4I5<2U9UYy8"
        "kvKd(Js69l@#y$CjNm^f2PZHd&P8-~HjkqDnK+yuL<eW__z)0}55{MR_~GS?51%f+{_*F#uaX0vTI?cEuZ2^sgtG=Vfq?UQD%MU}"
        "iH$SAqko6aIf(iuf#H9LKW72_&u>5{U>qkdk}OP}tX_*Ei9jvN8&M2_aRX%f%NMa&IN}!A6;*-Kmp}q}QNG@!0wp<^D0*D?V^~Ei"
        "Lad<wql6`jvXZOIsib`{eH{YSI=>FnUz{v&h!F5XDjOnQ6UHe(4oeXh(aI}a!sUJ2``hJc@9f*~n?HL!0Dduw<lI73b&*LF!)1}z"
        "8*e-qmQ|5#yulDuI|Z3|t`Bovci<!owVFsv;e5(7VVH0Yq%9H=2W4ItkqC<Xru5>Z0A0?DJFuEfnpBz*;&ub?5qK5)<|#r&hplj>"
        "yrhXUIWG>Y5XFm1Z;%h1H;zYO&TGdVqFtr#0O=3oV15T0=c)a|29y;JVded!V-GP7hAVL!Cri*wZ{Ymf)FMw&Oxy|+*!Dlev=-+@"
        "kr$r3EZ3)3ddG-4SSKa1XyO7ba?gs@zz-n(5~&-ec^H>onq*=C3RgJz%SkfY-;ieE8z37g#-Fo>{#0Zh;<5=P&h1Vrfj>g4R0~|j"
        "2zVg?G~j5Me3^!S-c)<A6#{qi1^B6TU4>wG&ZeHHpuYcg@%hv5zet6-BYB9@nG4{9aKPqQ;AD+<D=$|UDGIF+f7Ku*B#G2PFAt1u"
        "sYDB4xW0nl9ynx?Rps<+Q47?&q^yGciXPPrlm@ZFcRX85g8@p2tL8AN#2SRe_#Ojw;4mnJOc{NB3TkG7SlPrO(PiVszuhd%o)hI6"
        "psj^@3!DVX7VfI5Hs$2an<Bg!F2VcN^STrT!UKnf;3?kZ8<B;{n`IHk5CfxCo~L{0M&I1)So0tU134JbAgb`2d!pq*?bIwzZq}CN"
        "5FT+Q5B~L?Dp`xXuBOMMffJUXh~)+Xp=fbJxdA;&hVGpPqtOWdJGL2%_CiMP{OV|skpoix2}kglg~j<zhUP8ZL0$x9@?CIYb4lM6"
        "phC(;5dp^B*5MV1SU}PNo&tj=$s0I?ky-tI1wvotpwA;_e_v{_qc9FPkTQ@u<K*%e41RG<iG`KuD6Iku%t1C%wICv;qGU$6kn2DY"
        "s?Z3$k?))ikcpmrkMRl_ZHOB<Z{?i9E?x-%gLMs?oQvSQC~{<+NS3Qgb57ib7`xDgy5F*I8CX%CWS+t8{x@n9>}cYkI|CXn-M`c;"
        "cZPQNE&Gk4s{(*<S<fL>zI{7<^|xPtr4Z-Kc`^#SGbS};B#hzi%_c7=M7eU#I>uz1x&q-W^E5_>Kl9<g#<}E+MJXy@BNG(VA)7QZ"
        "<{Z;I@O5S|#fihEs64Z~fp1<a(rp_cv6V?yR$&J5+#rQWPhvy>{Z@y6{hm#f$mO&h%>iNz%wGl?I|{Z-V<8p-JTqwCbXzth|FQRF"
        "rX?k%BRdk6WRO86anQ`~At{N$529rh<SuFA1Z6b2mm-{u4&w*r+oarw;Rc*p?Ae6e+mEELnyWz)Lq2Sl^Lt>6JQV=sV&>=e0nA|+"
        "t94jhAr#Tt3)ZO2#AwzQ(hBJ-0FtHFeZUklZYig4jP$iKQWR42l9EQ(9dvQLwyd{rTvOXSw=wMYo*&F3TNMpci|Ky57#wX&6QgGj"
        "c7nuB-ERjJz3(AR%wypVtmtAKy?TRXoi-|kzJ-u790Id4+bw9JZdr<A10`$aU84m&j@+<^64ba}Z%R22%pEeY;7Z)-yh#<nc~MSP"
        "Z7{)fTklhO!+x@4=n*oH^i(Fpk{Ag7z|+3aCP#FpC!b7OFQ5BTC+P&<Lqt2LLFEbFLCON}LnjkAs{U6in55C|ap^v8VUgToN&v)I"
        "VcIO(o2zwqE34sw|BQ)Lp!1oMUG%@c6_6Bbx*0G_c`@K;;ad2zPQe)Oxyqx-B+ge{fMDaft0V?P3CNQ4Fg^@8<CTmc_tn@3D&awY"
        "(ejY43g}GxfD*{jvl8-|AS|OKnZ6Ifg$$(0yF4UQ*hFB^K+yZCTUU#{Q@3OO>o7|ez;e^+ptS)L<gVyC=gA_E!gML1_ExnH`rP1p"
        "%zcZig(I#ZI+ByrDRB=JB~t)$4<<B$YOd4<Njrd-zC+L3%8Lh%b6X^EN$@bj6@Ug}vCh$-Q@R%z<prw;aP!RPXs3t>FDWJBZgR$1"
        "FSTOEXEz&0nNPHXYqWthfl%Nt(zUj#^(dIl21sVBJjzpCrKL!_1}_&>5~|Xmf`ao3lm>uFzuI7+4pE(=yW0j)LGe>=8;ph2Neaw`"
        "5b<3*O1WzR8CQe=I_D|GQh-n@Y&ZsC6xEOsb>~sWD1%f9IZZ%J?4d(*?!h*KOP9q65bD&585&AijuD@$M@TTPWz)=hH5B>{464(8"
        "ZMa=KZFi&OOcOMyo{|xR`M_6*I!bFOKv7SWchFHT&#P@+0p<l*H^!qWJ^Q0zOWW(vt>Xh_4HdQ_q>)Q|?wWUlTA4ORE`Y#N(-Ou*"
        "6W5v<l-(;wMYZW;zRDq&ZiLxRYpXjIArMh(2}Z^)wVR#|+E<AssKT8)*g}2VYtrgX-x^4^!O7Lht&+Ffy}boBhMcVRs;twBYO!v_"
        "+u8r%ez?BLe6~H6vn*+nJEdAp9OP`jRQ_^ysfTAfmN5u#*iK!%W(CG8)p)dVw%d^QDDO#jh=0r=tmdVSQ*|=U%UNHsMWj9+ad1O3"
        "i_3hv#aR=_W~g7-qJYZHWAFo#1rG3+y9%Jlzk`-&n`%|vjfPF9j;$rsFA1}|Ru0q@@omkw!SaIUqTi6*(2ho`vZ`7zM+^%<7i_S{"
        "i_c}A*0N#HBW}iAmNxCk2(^-qL1=(CMGmoxn;{_}Q`FTLrEQk)_g+Z|-(m4bhXFi+)0IOfC65z%Xr@X2A(c`|OyjqhV5kLRaG8N0"
        "(I;zGOIx2H4L7J8OuKkM;!#^88GnEol!9`W=@#1tijOsx2N=kREc4E$iX#MmaX6(#CWIj~7;bFDj&}(x$t{#YW@5nB)ccy>!B&yj"
        "2jkeg6i)gajqIeg7{A4<=)NbHoFzycBwBW~mpZ%3@*8S~#Wi@$D1;2kP|Pic&|aFQW|D9TK<79pEp<ffSrM;ng*;3;eBLz6obSn|"
        "WlETIMwP4&w=oV8ZRMQWhCsOx@#NFYe73iyrzwvT8Sf2}{>~n(T-DSJ)(u9+Jp(FIj!R`BG!^~f=YJ?C3qQ8vUy4kq+DV`YE$p3>"
        "33I{<b6X3Wv33uv+oCnFz1`NwbjwU_I_lJ)YpvPv{^`@}Ev)gnG`YI7s)A4#0=BAFR)Xz<sJsds8K&(nn=+jm?KNF`)mVhwOwDK!"
        "++}u0Bk~PwKcsY@qBXdzWwi>@WSvxP)sSLFySwDdVNIDiucyw~%v11^b?rfAtUI+NmGVxM6yOcrF=-e+QLJh;DtWWx8Pz6KxD_Vg"
        "GOnpM5F6HvpEQioZ>+vRgSrO}63ng05!A&#LR5M3$pTRPqdqPDGUcw$9HeDz3eaiDQXuCoHzJ~5mnue)xm2m+k5`y|+y-hOsPZe3"
        "mDByv;i+}ZMu*w@UI`#}Aho;urRm@~G%xv%!6GaK6|wL(N6)M7mK1fC;cXIj&W3(q?7W@!P=Ig82jkJn{tnSo8=^)ruVc`4R+?>1"
        "Hg$e!*7%uBbN9?;ryLTtgPzB10|lu%&xKujsh2-^G^E3c#d;JX_bQT#3L>}g6rk>YQ%<3eL}tDd#&Li&CQz6wMWWE!J){N%m>PBE"
        "z9Z#&QVt5ZNb?&`uu@Q#^5_nkIvM@5l-H!ndIRYQsHx_j+R9LAXe;DBi5L5{>M*upiuOzy;HNA>86G;o^e%{2VFeKjq7R;mnzb(P"
        "o8~9pR6sRTw-u<3iG)0GccUdYZMVZvB+PNksq$!pi`Z%oT1ZS%y-02wMbNv@ilFw_s{*oBHL$E+p#uYxzb;2JG)j$jo|kgAon>y*"
        "1_a;a;GynX-8CRjA7L>6DA}QlE7c~0byKxv@|TKSnJ&tQ#ykSNxuY5M%GZsy+aMD++=H@_9{<WlDdYrd_ZsGKbao6d%nkotQ7$iI"
        "svAa<MCFgOTpq=Pd#SOGKn^2PNPI*@UdlLNLc;rML%NZrb}27Rs2fStlWb0F8a`~plVmDy85-5osAOAM?N#4x!tNOgV2G?oMZkY|"
        "DVuN->05KZOKU(A-~ucH`{}31IBX!Pf7%lA!1!Wv5mqMnDB32}_M=6OZ@F}S(~y~7r*xli%WV`KNb5xYRx<lo9?+wt-B+wxVt#6S"
        "Lhoitx3xSsF;2hzMsB9{vSnNLV-IYvJ#8ECdv^uw!~N*J4YfOBJfN~!-UotzFZaeXN)^VwGaeoL&e5n{$w-_CI{{!bc9#gS*}k(K"
        "lNopG_s$WIX-iC`UeO_E3yKwHlg8gFc>!fmfn6n=eiz<G)#T?`Q+5YUoiSm2WbgT@O^<tqgwX>(R`TnXMY5zT$|*V<2Mj5?+T=;b"
        "{a;vDId}r_j5l<ts(hnwsZk$8y2mIIW!Z9t(m~-h0jL^6x^}007MOTR_c4^Wl5eZLh*tVt*9k(rUL?!L?LoF@1hj1qnMgSV8=xa^"
        "><Gx|ZKp%~5Ab5$wFugVMG)np3e?58G5{A6G)~zBb`^lsZZV@>>0=~FZix9qb@zboa<&eO8sE(Tn@b6F#KAHD+;s)h@t|)4n%fF8"
        "2>Se(iyfd5Vd@q}JTTFnESn2NLYn8nHVOTBh<tW9B9hv1<ENuhFV@|Ik(RdYf}OWe$EhoHRL~KOcC*wbcveBjGFQ8%<GoMP0ERU9"
        "*F{9BB^ldRf9#M&j-&!jr;la7_AqJJYqx^exuC`oL|u@%yjQNWB2))Fir44B<Rq|mxQ29J-ag#oN*I$QgDgUg(^UhSA7-FeW0W&u"
        "V0LY#$hvgVa~U8ytAE0y^(GqIR_#cP{P-}c<8WAr5-WQVLTV}2d2ttDjB8H!xSx}Ei*~+hufi@RNhA4+?uT=fbTiz&?YlF8_rq6^"
        "Xr#O#<p(EyHI2=W9?frMUecAeI;TDE;_>9HN7s8MJZV_FW_4X7;`v<LpfHJQ4)NPJLblu$u~A4V$a^3^aDi*ki)E44<*I#lY*fW="
        "osx{`h67vn-fh6C)2M#?b{kWQcJMIp9a@rWCPYV!U-l(2liI!e{PD~0=U>nN)9R5@XVO}b4PuN$JK|iNY~wDU_+$p_8b2ZXAnJjG"
        "*s6z3;zt$IwX{xyZ&O9r(liadP5rQUL*1Rc0L3NS>AA7wveP?rS4~}};5nEz2YpnSqVvX2Ht2dR%woO_cdVRP1Lt*dpvx<+pG2X-"
        "Yz*2-M&P<oSsIV*gbVy4gFFv;Z=Z4l%4$LSj+|(z6r}t3K}^pjH=>A+0SZr27X&IL4}wlr0&8R?-W>u6n4|zb`+aBDa#P!T$G{qk"
        ">ja7|R6=Xt?K@^xn%jRvOj_BF_ruI*F-g|s%XIdEDWCq<cKDlW@_eBYFBWA~BpcMDX%NJD1k6z>Z#~BoIszL~hpwK>rxIsXQ<uKK"
        "%wqf|MEq4Jppe`}s0<o#z(hhIXM3(1acw{X(Jd&?-P@B_mG+vuX|PUpw-Hk;H6C5EgGrjg^XJQ(qiQ<kFwrUpA(vD6oe$=Ji?B|s"
        ">F6;`OmFx0=<7eg8Ilc8vGj!?PysRUawXEu)O}AM2yWyD*!<H#X&2H?RK2R(&sunR$)Sh!^xow_oz3dX=V^9ro!?wScqWQn<@Hqk"
        "eEvUQ&j0xE@%+=*;L~p(&o6Xcr4yeX2I(QqBmMjnY}$%~MJ}UJZJK`&r-~|jEEoO>0<*`EJvCcp(a)USq>p<t1U`QkyK2*uY>?rb"
        "y8&(Xbn>~QWVU0$-u+WpAiC~BO0uUCl_!Y7M8G(4`UkP`ue~W(P@bGl_O*Am&k6~Qd{!$v?)&JOfZLyBu!NQMUF@%2dMEn4-q9bu"
        "NJ#p8k{5*bRbKpK;nW_$zn-to$UQg4Ei>WX>yqtkpqZ?{GcY7>gZ$B4SN}cOuNVh(t0(P~Y~$OQ2)m6_jlnv`+|sY>Z9VMLnUnF!"
        "YRb_v)<vrw+H<W7?#!C;&%V_6?EAfXMUCrUalLmzHwv072fgKH=TbDUMEfuH?Rq`cObPUdBb~!Bny>4=-ENw;C;m(wtugczJ~R4%"
        "5-_DtKM6n4h-pv%e@NGjj<$>QE2ux@H4-0w;I$IMYkd(xmr?Y!o`-=-|6wGc!XpUC;DZ2*fgsQZFH3`$Fa8fuO9KQH000080000X"
        "0Q>}I39bPE0Qv#|044wc0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFKA(MZe??GE^v8mQcG(aF%Z5R{2zurHQ;X2>mCYQVv|5f"
        "LY%foGnQs;VLi-@ls<laN4s^L(&!)<N%Or%P17u|K6QX57xch%M8~jWv@Y(z#Y{Y)p%&IeV3U!{<+uLsYyW)vaNh>oHcd01&1VCr"
        "07uMw@3!E=F>!`REnn}pZb%0GjzYlJq?lP5<@9v<g1%sODLzp>c&|we4ny+x)UzVYh3P)!Cqnyc$(*>-&1c4=2&}&^zfd&$8{|kL"
        "-FybREDb;t8-19L>Jsw90**}9nJPmz)l4^fSb)WhLV3gV;zVydkjx9<5;x9{#e{SLA3y)+r8~1|SgTS+!*wzT;))Q7J6QF%cbn&j"
        "&11h_FQD$H3NWhUI4~+pF=I1TtI}4hC*<o%;iUpXz~}}NO*a(DI#QF^R)=IVS<r_E8DY0Io`8JNRALG)VouziL3?dthu-~Gsfjh7"
        "`X4Nq6pqMFQdF?5CvXBWWk8T(jAlI)e+pS+X#WY|TrfJJcgOf<Fk_eg&}IS6-_Z8cK!JXeDV1O}!z9IL)xn`E#_2-Ss>*&iooxGt"
        "`(i%(15ir?1QY-O00;m803iV8>g^7s3jhFcB>(^{0001Ua$_%IX=ZO@VQgn*ZEayMWp-g~bzyXAZ*DJcVRUJBWnXZ4bZBpGE^v9Z"
        "8t-%4w)K1N@;{*LhmxzzI9t2hsZ~1HmTTQ&$!DqEa&C?XlaR$TMX~^BTlMn4?>zt{K?+ooZc{E22|T>_@P6aLp6AV}OdlAz+R06k"
        "lUtU@n=R${M8pM8^ScpYPi2<IsU&fcFnD8eCDS4wk#+jS5|XgI*rqv^1sCDV7cXC&5K-Nhyoi|)uO3+bKvF^SLJ~>@tmUcP5kd3h"
        "ws;E3iU}zQuX0l4*^WGJSWY<mt%6~e06Nd}5NEnA3l12BY@Q3zJaX1NwxpXO$+qP>&03hMN|To_*1Xt~gi0D`R0t+WLnBFJIcld2"
        "Xi4QJOK)qiD|kiD)ZA^lE@GP9vF(<Iagj?-W7%N5W@RB#kk4+J(^6~-=~{;L)LQl=>y<Ihi#!F8NSKw;hKkMLa^NxESX(K}Y!|7)"
        "NEQtaRH@=Hs`7N5vLs5<^?C$KS+j@(v$#ZU?&XWWH)8OC=I<<@ta!zOmoL;9(Z;i);{KY6DwE@vF9>|nTu7S7EJ~9x5t8ds7?YK-"
        "1~g~LB``52w?&cZiFK97Xk)PP63FN&@^3OPa;70TqoT+aE;1o#B$`4}RMIiZIJZafU&7YP##vM<)7NtH-5VGgOh$rZhfWPh$r;*D"
        "YM3*l66`q=F;7cr5C5rNq&d*OMQ<}^1A+gs7`&mi0kNTf{`Rjn49&9QF+vB88V5B}n~-yw305<bZW)Lu0#7XR1SGr83Mv8UFUNgy"
        "hN%a}MZFuN0szM!SP+n-H-kp7?o%V*Psp*ERZ66(VB{B?Rcy+6!F{h@u&qE|x1bsnG9dSY2IqiJ`1M;rzL*Tl>=1GG0523p*F_y|"
        "Q?W(ksJW1QXCcyIa5Cg={$aQCTZ0LLS_3gFCGlI{Mr5%x4{d-8S6W!pr!oZ`0Yoms5XjynlNHbRG~`?MJtE%K$?EhXIzO4cot}9k"
        "V=wg&sNGj+gPZZU$H(tmO{kdeGQ5w|Zy|c#2HY%HC#%_Mw3xqr2h3o&8EJDPb*^eCkt7x9E=lOM7`qHwo)^Ma($Wt;TRyY-`SnRO"
        "y}n*t?{QSrnG*>SL#Q^CFUe0umgqP)BAYaUWT^lJfdoimP#eo-3W6p%14kq92H$eS$QDGJfn>@`h8o#pDmP6s3s($dJmtGHP=1Vo"
        "2rUV0apIZ0aD?`J<SDzz>6Y2sE(-K{%waxG1tDjm$R3y<Saw}-f~0^6ZQ~)niE&qVm?{93uTZ|~<q%)7icf))grz403R3}VGo=i0"
        "A%!#O(fg?DBBF?in3he{uz3Y;I7m6%-X|&d^^2HjowBDCjHS4*bKKtAM<`xcbh`r;>aD4K4cMS+0)6>jCD%uP^8#zFKn3Rv<FNPR"
        "?0j*0@-~|O0$=B|`D}SHJ&Qc@wJJ$M{QAZZ!VP<})+>vG)keU+7a9FVX43_CqJSYUEO)jHz9j!(tW@T;hKemCSm=ShRXL>7c!Qoq"
        "=o&4M=7doW1sRA$q5)`g#7%}aK8R!z?K#}iH1`pBqQXXiHQytLkO-ks6l6`)42sEDV6<5MRUk$IktNMCHD%%K##TMTFylb&hq@EY"
        "w_MuVoQaytT`uX9G7DEgJmmEIhW5!}V3va}(u6cBA$77x)wChBOTv1s1sS0`3fS(mR}Xm~S}+uI10tC0`)iIRI;abU^+>aOABER<"
        "A@I0V)LZj$$w)1jsjAWCa9!@gs)VYLIpn3VLx>8gKu7`{saN%~HgCuu5)Jh}kedc(0$$SWdmsOP=s2*EbEuI66=xMSu;41|QE+Uv"
        "mGJ+0n@GsL#dc5^**_b0FVV+ve{(>);xrdPYs+|FV_4ZffM8^F3hCJwEQ8`6S#hZrXTv*|GoHqgTBQUTIGjO5$TJ4}0g-fd5n>gu"
        "6>=gW^WIj{g%&@Jl5j~#B^K(#BC1!o=!NKj#ti0|**=A!S`&i%!tT1J?$8fv+aKPlBHF{RF}IfG67IT|uR}-pQP}R;?y=wrI{(eR"
        "vb6ig%$hksbstiGhjp-rA=!HmG3dsPRtW>xHmiMDH_?kmM2TgBv!{jHt@L$p9}wz$_pk}Kh$k+}U6euc!c0A2MKwptc85XmgTkHE"
        "Lif{T(&k1u2FLf2N;I&OSUxGZw*^i7AQ&A+3o;Os8x-pMk3iNkpWGaRJVZ|gE%&rT%0`s9$PN>-i3N#ClvS=%B|by#h9?-X*t<sd"
        "!tVr?Ak?0(vW2bQ4dpo;HTHeptNSfG@oV2l*B*?9x}eJi<I+DKk!MVTs-XLVe+c&0co?)?+JmaLxZ`vijQ7`Qkznoy1`S-O2i(Bm"
        "8U`N96LpdW7t$l4D$B2{93#&$QMpvsbB|Oks~8Vm>nihySW)b#(sa2K5bLMlM*eHZUM*D`uJ>V}z$r?sd+xI07QD{i$FuhEAmqoz"
        "wuD;Lf40m3WIQ5YeXma(M@>`jNNw`XS1^Z5zY0b~({Uj$c?ur?To<r=YO{-6eeBWYF*dM#V>nP#V`U7cH>qo5yL`M%^K=V!D9ayG"
        "UgTRK^&QBaT%5o9DU}!1?W>ckS?8F9$u-|FB$SY=cdLuV98RwnRspF_q{<PV@nC!5;gRy)ay7lYUMyDl_Rrb;Y=$MAce=P-UQJIu"
        "eFnam{!7ijJ&WGXe!M=peitbOR{e$DOh62NKi1(<2Vu4|(GsI^AwsjC5Hmh#PK9K)gMy-D9AERA%Pwkn#c{E?oK70==>?0Qm(%O3"
        ">%}=7-3MV-msi7c(?gS&S0JdtX)XTw0(n@@R?~_1+H-<joGdSv)9Km7J9Y!i7Ss9Z;%qkmsWy|5)6PJSL+z<9+8PSaO)L~}mWek$"
        "_kK;j{xyUz|A(>q4Sw)OLm2mK%?7Cn9<Gz=Cm$^o&d)wx)Et<egJCuNc&WU9tiq33k7tRtt;DOL)O?_Z*y~^($a&t;tZxS0>G<HZ"
        "$vkC7B?i0ibeyVR!>$$~+YccdO4;Cu6Zk(c8`Bb0%%Y-_WhHgvzzvCU@Zh{|-%Va0I{<>dJ?oa>A8Z5GpZ~VqHP8#JKQxvV#ZE_U"
        "m%1Nc_ucj!ss__+sp}<^cO3>;BX66#;T<n3z>ACYEKs6c3hb2mdr;U9>34*j-K9V5!)}PKJl!w{%*1rsT~lk%!~t+wv?fwFs>f}l"
        "c<##q3Iij`XZ`yjba7<~YCm=~Mb}5e#?5|U(EhJ%$ux0Xo)|wS6HGdesbL5GH0M0znEi6a1%eJBCLU)Hb7JNKsa?Bw|IvfoZ(a?0"
        "^(S_)x_-QxT}~H2uNogV4gI={t{T>c!;WJZW=yR4+%DSxAeP{rpUK^-{I=qQD$enA-Skz{yav<J#;QuyeIqtruTGXr4<nAj%s>#t"
        "!wU~9bHcQDshULdLY@~@o~ZqMT?)W5#v4!fm<AwvqG}nj#nGXxy1QSvTk?Iu@86o$cL9;r2S<!>kfQk=OKk0a{*{btGyO_&-<xo#"
        "FLGa<xnQuCNKYghN;esL=)At4b(9_&XY^~c0Mq+^4W|pSVaXdWxH<af`1p9-4>TR6IbZ_os<_(12`c@aMY+D8&})Tne#azBg}mvD"
        "^}V_w#S6anRy1R||E!vUmXtevLxb~!y30gxGY}lSzf3CpLk&1U*k)jn5VaXli58>FV;j4kZM+Y*Y^K8Ag*;ZkO26yo{A{sp@Tkl6"
        "kK@5~ugu~QE@QOM?e@K96W+Mpkfl4%qNP~@K>c7t>D|JVei%T2lMx5eHh|=nB}Pzc`=X+g+55yASGL{D<{sn)!T5tVsXs`?iFjRQ"
        "SrZSsLLzVK7`o48S2XpXXY6XOlq$o{-)(5U*($S{?gTl<sFQac9hd-u&#vH8^U+2T^R&lGucv|cG8z(K1jKKa_|FZV{^}3b@((tD"
        "W**qP(SI#u%sX(~a2-+Lu>T)WO9KQH000080000X0JanEsMi4i02TxQ04e|g0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFK}{Y"
        "WNBk`X>V?GE^v8$Qq6ALFbuvM@H-GX*;rx>I}IMP(^6o=&=u>j9lFAhB|2p*O9mw`1-icbEIVzSEG<tW`TbB5g%Inv?@Dk@LyDw<"
        "%U|ndjaJ>jm&<SGUtvp(&^T^o_y89FAOl%tdX2DgMuX%)ja0!GLWnd;6KXrh0dBpsY0|K3K{;bk1#;FGvR1G|d=AL6Hn>A|GTa!d"
        "pSl-A8s)}bO^s91Y_M%*p_~mI9iO1zv6I9XjG9K5NJ|rq4Q!bLR(^})jjHY$;<MSgd>-ZMSn7H)&Q*dB41dR=a#((Zb7!$k6BvTs"
        "*24;}uBX%^+9CBAvV5uxj|Y3V;q_qdJz-){GiwxNeUMgRMLJTV><l{^OY#UmrRj0a>^N)0r@k|!WI&kFOL#b~`zZ3gVl*hE*4b(7"
        "3>hz6$gV@Hv%`tvpBK7sJAc%O@$i2shM1X%@`V>zyufXpMpIeYV&%AsKXoXv@4v1vFx3%tES+FIo~WVqJ!2)kB3hkE<B#oTueUn}"
        "$r}vAqQsTT;^YU7!Jg5EqP^3^*_8Oz5AJaRcuzi5?ruyES<}7;ybqc9ZAEeG$Y$}sa(Dw{DQ?-gu{cZ#Lga&<RZf%LtVAE0<@>l4"
        "{{T=+0|XQR000O8001EX6y<DOKmY&$LjV8(CIA2cb8=%ZVrgb?V_|G(Wo>O?FKKROWpZV1V`VR2UukY>bYEXCaCuWwQt}MZO)O6="
        "N>wN*%1<dxF3HT#Q^?FqOD#&xOHNfNO3h8o%qvz%1PPX;rWYlG6(r^4Cui#^DJk)C0RT`-0|XQR000O8001EXDv?LZ;0FKz<Qo70"
        "C;$Keb8=%ZVrgb?V_|G(Wo>O?FKKROWpZV1V`VRAWo~71VRUJ4ZZ2?n?OENA+cpw^Uts?SR!>r@2zJpHSLYPmOIj2M1c$WOmu(=>"
        "5@i!Bkv@{z?DpFKy)*ofC|Pzk#a)2|&UsjyAM-uGnPHM7`F^t{`Y4ql2gO+PL=<b8Vxnf`zywp&a4B99A&FuANh;crzyCyPS|1sa"
        "inR2e5nW5gbde;<?al2?t7K0aYG~b2t(hhq@0n={5BW7CRec;U?RhI}+8tQm)1sE5<%e+j&EttF-ZNo-II;S|d824;Mt<+uiE{Pk"
        "k@lyK<^JbC6l-{mI<3(_5ep;lnc%-z#RIKYs-Bt(%px16D&)l;O-?#Vo9Y21YdLEwf0L;-+PbD~E4wC;{FYHODE3yGcZ@bqZ<LY>"
        "EpdDEx6x85kokp)@<%nW?Dod4k?*XZz8jshyS*XsvCJ#p=p8WwXne;72v`)wd$+HRJXMNO4L9J2a;rfjdmFC8CAZNqw{e1a;E(ft"
        "{MWG7M@;38rtV%3U%Pxxd`&2S#FM{r;)Va))5l7%53wYHjnu0INJ-LdPPUt`bMn<Di|80>XjcUlBZ_e4%gr}m!_kbqe0|B(5a@5L"
        "YD8k5P|-lv)UrPfhCyl!8j{2lClIz^ZFYJ^^uP_>cMOAxK*qooBN#aJ#5-wx`XG7U5^v0Ny~$t6cG?3%LYl~kv#pMablS?u=Sy0z"
        "FB-$q$#xSxMY(3=C)y3nhH9Ej@KEiC=D;959&4}{Ym&^}dw}TQ4;@ucU{@O$g=gclyRauJ4LLKa-gA+rFQU-N$%{yf3@org?LhSF"
        "=U$hQ09Q2`^A>MbjjX_^Rm)9fWv(VFK`|85s^i%cuE|>|Tqwj*e~s&&YSwYVs+tO92=T!91FPmK8CW*#fkUj7L;&!lj{Au!+B3tH"
        "PP43VCq@C?&%!*N;8sZzI7pM{I#jWL4ID7N0@MR^qa9eMsXj1%IGRx;u=WF&gSKWGJq#es1s8`14hr@_yVT#3gMmDO<98mI945g+"
        "0s(t4$ExE!H$E54%yl2QIl7uraN*08<(|f&>Y)cHx}yfn2>RVcj(MMx1}#&%6P*-?+*_=4f3h=rU<-Xe5(RH-ko2CVur;T)m$F-2"
        "*9PpUM;kdwLK$%s0*7ohx-iXcgs~31hTrB4E=A<REDKf6>1c4gE&e6xxHd6<%nu@nRREa~N8-TJIkGIhVI^$znMVS*Pj-h5t#x-;"
        "B>0Fs@Z*pO2mo?XW0cFOoFs0e!aS=wdEgN6`3$}M>IzM_0W^I{w6dU_xsQOz&3C(;yxhKDF;r{4jbf)h)^!UhWmxJN7qo+PS~D_!"
        "&J$Z8*dl={N^!5+iWb&~paj%I&#|EDfOz|In-xH&2j=J3Tw)jWbYh}OEs>Q|pq--+p%V)C1wBSwFkp&mF9*>e22Uga*1p=fj7Wf1"
        "vQB3YgznF`LhP*x4Aue@$H)_5TROwNEPxm^bK<OFNNr1!I@4si*D1H9IfA1sUl!|Z>avt=T%i$z2E*W{Zw@px!VpiPznzMhZO)+@"
        "=EMVx?f1y94nSPi=eG3EOHGNb4|G0Lm-CiEZ;L`h?iz>zsPQLi>f<Z$9)fsJpP=7SP~R!o6U&id9sC#?h!SjyjN*v>L6E7(@qIuJ"
        "7*by9qEpnc7*8$~91SmE7nqe7n)K9L$l|JB8|$@=9lv<ivts`^6?`j<-;DypuRKGM@#MN9=geK~_%(k$4f^JUHcf?up*jZ#+BqSj"
        "2VyTH%{P6EzJHz`=Liyxf4t|zj|){a-<)pp&Si-d7$js6)=s41ZA;prYk3F#(192jExCY@9qgb+1|3#IaSP$}vq;X$VC3RZG#91Z"
        ";>8tF%#p&x#~-0!n<J3nS`X~GQed0yBd)L)eOLk0-G9qMAi<wt3FZv3WQGcdp!7W(G7U9QA-;A`RSl#nd_w2|BO<Nu{+%e<v1CV*"
        "K{ExVHnDOG153_KzlBlI68FhdeoI06XamzB3Qu>{W<(w*7*1dnxK-SPA!kIUt`;;N(=TY@)6b3lE2(EGWfNRXa~TdR?%UnwJ^5@h"
        "%pS^LnA)E`X~CV7?07xSBEIs>1|i2T51gWj-BDr=#-l>ATfDG@(b`4wZt<A%v59<~*|BPtw%$aK6chBtTUTn;#BSxeVj9`Qp=;Cq"
        "U$)<T5vKd)i_mU9+>J#7{I`&MrMq@sTYJ=5rug@pEk^kd1Lg%}92N~oH~|iRV4gM3k)e)E%-yqF5hveb9z|DTh*_6DGo;**i#MvC"
        ">mCCm#FiaQ7QqBYyws6|V2TrN7BMR!5bgK~D8uX8{0(H$tDG;*(>O#rHU?|&RAYF~&><Gx>wW1Zrkwv=oLh^*^3uj!a##FE`HS&k"
        "Z}c^uzzBiAcQ8U=D;+;{MUHp0@Z4^kHEilf_#DqF`ktTGk!=Px%gjl~iU&oB;!w_^L9tzrpglC3mS?ttLk*_~A+_OkWt+9AwLO1N"
        "j7r`dG}h7O>H}iA-F)%oMPw<T0V!p`lHGr}rFCx?e}3zEspTAByFaPp{DYll6cBdCs3a)WcJjk&gBiDppOfTg@lPO!t%^o;MetlU"
        "YsvL~?2I7vI`CySBK*I<`CkXuNX!+Sjj+;(^vWR%z-yOR068)OcH9@CeGamphR=CKHu|rg+ne73P)h>@6aWAK2mk;8Apmvagc~UV"
        "0086y001KZ0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1z?CX>MtBUtcb8d3}*hZo?oDh4+?x2gEwc4Z29O$f{D6c2`vh!^EZn1`Xq;"
        "zJ0M%JN;4225H{oW8RFEa_`P)v_C=!T4*wzV=&y4_9CF|J>-orJrP9qgJ8%ER$Lr+A`g)}U}s7-W~988Qr1;noe6_zuH1IyZVL%~"
        "-ZtcH3&Mqj#Nwz6>Oh$DjthYu#{FbiA((QhmUQY=B785MBF((r^UHBmo@QZ%1EB$HU2OCGTBqa(bwRRf5miF(%&@`<KrL9M=KwS9"
        "1ZOBCGw?VWmVhg$sM&oi!)bcwt-96}yEv%Hw(_2^FTVEHLBaz@hEF~f1!yCt+m6Csv+2fFDW$zvN<4_yx)SBcJ0Hu<w|;kPi{oDQ"
        "#jrB!#q)2OKk$Eo-IU+?#`;@beF0EQ0|XQR000O8001EXIFgjua|ZwbJ{JH0ApigXb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVVsdF@"
        "XJsyMd97GmZ`(E$eh-lUKx8jcD=U|^d+^`^y=5)BbzPca1BybRCCcWkTSUo8`mo==?~syA(Q?*qGlJG455IGGF5fxw$;rttRU^xS"
        "XH1ET*SrzQj5m!aXqD9>6}2d0!P2s3e3@}tRsk!@hBa%!5|Q#YYZ$NNwQNM(w6zG2kB*P7X;ao%nSHptW(}`bqG7EPisb^|6SmWm"
        ")@9CQAsfjv`B|%Ao)Q2kV|DuM>c#6f*RNh&|915%%oAV3zv-ocw_*!`@nY?rg=bIwfT<We#Y!6(*QHXU8!4HzxbE_vk$7lKl||Jy"
        "yG^NBTf}Q#z|il3gXg#uX)0q09NH!aa8`*r#sXgnR;H}Nc~Ngga(tACxJ*RNmSWAf628M9{HWJfHBSgeh3rZ|*^eU6IV@-dcC?Wl"
        "zRask<d75z%se?cp_^l|tQ(f|W_^65J=fe<blYyDkEbQul!(swDJ%-cp{3Yc4pZE`O9@C^63+2aoN=Yt)z2T^m38hx>gO_PGvUvV"
        "j~G6%4SO)U&UYb1uoH5#cHgp=$x3|R`E4s|nM!P%yJ!3!eyK&3@mOCt>@G{#NHG>glpaxptjN*;iIvaEVF44BnzJ-3d4nW47qjnr"
        "i|C_T#K-f);k0k<0Vu?29K*lP`mK<wH6h}n?@<)FXaHn9R2D*cFu?b115qBgyiPj%?&h64uSKEC+Gn#LMhljWTV6{}1-4*!VO8FG"
        "7k-!vUU;+ffNg}RplWelx5Bc$7AUiV-LXd<YOQYS#v7m>54ZCnV%^A&<&RYAFQ_`6^~LdA$7nLTTAFF`*&xr%0hC5jpwKA^pG#Bp"
        "CVyUa8P5xx<*mZfOH&wO@4RRAN?Gpc>L$_=el3$k6clZEjr5_$bWIzmzQ#NeuEL;8r6=swljp<{Jz`gN4<lA}S(TKoeT}-97bDD~"
        "&7-y833BgUn=c^`oE^vLElZr9jsBN|Vq~UX580cPy+cw3J(ns3)%F8BJ>4m^P!qDbZRTtjk8X*sv*b&e$>!cV&@pVZLM)>#&*rRS"
        "QAXs+XqlxvMoNVUli_t#F0mx@EP`c!>9w=7Cw{O)*TrKIFNdd-9ivhT@GL8DX@6iqD&i8HP<$43*>MC}|EEbI`?~bGCMHQrVd!zy"
        "qBKE{(8W@D2W<!Fc*Mdx_T6EFsiK=rMD#Uhzpn-CuPF+=s<Qi@^%xxdtG>ru0HWZjKQ%L0o3wEg4y7{5TVsL=4kSEL+JVd+Z&?Y5"
        "4Whp9f~kD#o54p&eS(-yw22hz85D619!eNX?rCtqf6KDWAVQqgiD>>K)nCKSb_FbKH@O^M*mBmBBR9B=XubdL$n=hb$naIB)_0S5"
        "bQLBtB@ULUgDb(0y@if3kH(svgH%w|8qUKrcB*@RsHuK*#`T;P4IH(e`pl%yH|+hc%p8E5h)U43zn^3wc<osi4V*g#Y$>@~Tyi+X"
        "HD$ApWJTy}khcFD5)ytN%RN52?+D9w;jeFJ@3dcsib?lEFn1`@IJgT=KTsWtr{%~mJV9yspi12NO+x*mx%VE_<On)mM7JxVzTar!"
        "m*&Ci79Cg70Kuw$mYVOzebBRN<28B)m_;`kq8lHWF8$bWrXAh+nv0&Iin7A<&N2C5>6GEy4Bz}Ql|`#TT31q5u5bpG^>|N}0m0U6"
        "f6L7$-8Uyk#YPW%-bbVS6S4}Ux>%S^O{L#nOplTO*e+B9t!g}GPPC3M6wA5VSk1zgcDlwxqbz#grLiA@#)L_)jPC1~Y4M?*8_onc"
        "bxWg)db(QkN-%H9n|Q6OG|EIVDp%i`8C+K!!h)Q^vS{TH*4SUV<6&f{=>k&^(q$_ciR>#m*MFTVr~2pv1jinq-C`IKx}Bb$BO4PL"
        "(?#&&kjVt0*a$r)?Xa%&+kxv+Gr+f1qW|wl=iVt_=T<%B^bbo*KC(KsfetnDPCBw29a;T)EWMMz={u(BN!}RU^5CWyG%d#)16zXI"
        "UT}Lr(59IyVLXFbuy@YTx~b>fn43)SFd2RmZl$>OY!8jQe=u(Te%N53RrDs}Ax$ov>um<O!G-@8j-0Ul?{K!Op6OZnUvxw(#I)<("
        "ICWt~R_||I#RIvh7(=-=%;+9v7J*?{3(TLlnRkPiYwJGrvONPA_v4tJyKFVMhF$#jB<NhnkqqHE4{d6}F6R67{tf#uMM9cozmzjZ"
        "b}dGaALl{$ZfX9VfBFYosXvje##h8^6Go5SYCVMnB9bVTS?2v^BM0Wq!8@7BB0ce)TTIL>59&d~w!!T<X({t#2OYW$Rhc$9zw@l)"
        "`uC&7?A$+iY;VHxYyl_#e%GsdjKSN_1WgRZKS2{3*Njl3{IvB2Y<qx~mOS3r+%#w!CfR9K{a3f5!?;8xt=h6xcr}|QuPNy6jUAyk"
        "_@g=7>$Ocxgw_uP{f|gxF=wBF@du)8jf0O^|65hz{SmO&mUcav>7Qo@FY1HCbOz=soz7uWeg8MLu@p?Oz#on8d3^LYP)h>@6aWAK"
        "2mk;8Api+!qYA$P002V+001BW0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!3IV`ybAaCwbWOK;mS488}*e;|BnfyUWs2OowFLkkQ;"
        "cgTWW4Mn13A+jitOgit^kB*zAD>`7h2o@jTgLG9@kMG_$P?*BeYorX``5xsCLKKbJD|Ac)XpcZCKuqZ<%1+}6K`?czuRj1Sb8t#r"
        "S5>uKEEngJ;MDo#L{K0Og;=joQ_}OBmHGZp(Jpr6UE1R?xLUmHx-;sW`hZT4jNeE%=)!o<OgvRC7aw1AK3{O3Jq6@P9Eitexd1z!"
        "vf@u|yO{ULg=5wROwAm{xuP}s{!qw*E#oPQnerkl;QwUxF$B!5s8f_?)>v;b%Z2bva)X5=T6Fw8x!znLI@qJO#O=)KW-|KjpY(5G"
        "_2Esu)3J#>94FTx<__5!+8}cp8#v%-El<cc9+4xiCA%U$>00T4IsQUkKVjVWD(g?*w@@fWKh_ibOY;q`W`^hez^u`5o@s*X6d5o?"
        "l^V9}Vm99MdV%vyrk{!}&IELADT+nF<zOsv)l)IvCzqlPc-l^90GZUV+%92&-DuEJ&Y<kP)_*C#ei*#qn+tR^{Q=b-R9}XYa4-?C"
        "5RJ;@|9Ylu>rPMEOzUH*tJ~$`H&9Ch1QY-O00;m803iU#MspTb1ONb~3jhEv0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MTXk}w-"
        "aBpdDbYFCGVQzC~WpXZXd5u=zkK;5Dey`O3VK6TV*A4K%OGJ5r10exg>9qIKT}5s(X-yM5vfb`2OaFIf9H&j%9(#U?G9J&*Z$4WH"
        "@xfbFdhqzk-x=M!fRi<P)XrE3-b(Fiv``xhev8mqB$?7!fwbjTc`W_jvfQjOo6qL6kFrsf^yok~4eap6ee1TeqyIIM6$GhYpfue}"
        "27ejQ=%0<<K}pIElqM(2L2Igg!{&q#Y*V$Ju^ueWXSFpgbkc7dwT39ar{9s9Hxyaa948aESyY$~H6NMJDy&aw7F>m*++w-wjM6@t"
        "&w$oimb<mlxCoB(;BNunEWqtlS5W+w%VBLy155Z-HjWi7){WY1eZe)I?#^c^-2DU<l|Anru(tX{`+_pZCTpbL?BxcF{#W!`UfK<M"
        "(i-B)oTow;Y$!>49sJZ$N2ohXc<m*;s0%q9WV6S!j5|v5MK2Gf5B4+aY3l}iyi^U^cu3CKQV%OW3`+_b+cj2&H#;hG_Kcjgh^LSJ"
        "=UMsfx$KGaHsM013#hGZkpZw2ofiwJ@SsXuin7~_6r`h;%L;9N4n`iNYUH{>%Fv-tdWDQ}mZYPgTjiY68`4_E>b?sR$Fw=$Q>`e$"
        "i4JE=`zKdVlUhW}qnLQnP&kB-T+8>?8k^MO&ltiPPY|J)!)vs%%71t|iuC{5GK4ua=E89pE=n^Nf;w2ZhKiVsQ%ZX|tByMxGxGbc"
        "!6fXa@YOQ@hXAr(dg1eC!q+MxAMqgsNcpzXm)nrwD}{8-!I`}+5nh9j<6;2#>$U%*xVC<c(CN?xkP_KML&_B`dKvs**EUEPL2tEH"
        "{H~Q(#Bu*1q1Ab%rB_CSBi5mb9Y}86^GGh)xXo&#Nv-0R*f6cD5y*JZx_*XFO}sf~N6vgSmPqUprs4uneW8np`xCD7D<p6g&&td%"
        ");k{Gh^7>-ti@n&Afz;!$}{F1UV%17HdYN$!eBjrx(exHSijg$k2ao*tll?G_`QKqD;oKN);(s)WA!e5%p5j#A-&fm*%frJ{r&Hu"
        "|Lh9tSxbNXA<fu?1&25_d@*9Z2a1KzO_ptPzEGlfGBRPWnEQ=*Lmt;+#yVeEVSDxcQwl!<7q&N_-agQ8t|sT6&IRC+*&QN8i=cdw"
        "=8a}dft^XhhOQE(ps6lH^-EZ#77x0WVI25!jtrXj6e8Z@fC~@dA_~ugOq*h3WpzcTyiOcE9`LuBcvz9vCY0QVJ>}%vr@w|@lqEOt"
        "^ZmnJOR0yq!YB}aJJAdq#rH5xJHek@Nz@6y-9NznAMaz4vfB<uwlcO=Bz?d!9(v&ItscJ~M0geTp^H3Tm5a%KxYY`>rm!NF?}jUN"
        "#KwV}7?ML3(_1DE@%6}7KC3*n978$_#;T}GdP3QrAB)IUbmB~!W9=z;8T>IWK0$9stdgn9BrHul#FZLyiK={d-zE&s&v5GKO>*Mq"
        "!7@Fe{sT};0|XQR000O8001EX`6tF3a037Ug9!itF8}}lb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVV{c?-UuSY*aA;p}a&Ky7V{~tF"
        "E^v93Rc&w6FcAJeY5(C=B$S9WbTB4VR3sQMkYFI#H-sWLxweMH4z>g9zvukY#0dpNz9hEqo_l<E&t*27J^OS6ci~c22%n7VO4BKX"
        "4@!Q;(r_u@ovgYVVIvj1m0~-;l)~Vx@gA45X%+5JXfn-DPfkyQrv}}%ZOyR)8$eiNvu_dJywJcz1$r+GyM+o{vjZ-Sq+P&fja9|P"
        "7Ie!>B)}VmIM>`5R6tP`bEGoshoVAJ?g6VU>db*&Vbd^+hFirPdq$TL(g1JT8XKgOicn{>ndQ1svazgeKr)cZ!28y^vDzMoEhAOA"
        "8?SOf$Qp#OW~V1*&9t`nyi%;)g$ziQ2$B2jrOVS3poilABR~I@91r?TZFRu6kLa4h!W9l~3D+I@DUchexHeprW*RZ^d5ALoLzZma"
        ")lBWvd_Zp4ZQ&>_;T^l>P1ks$RFcxU2f{?xagAKvx~O=wj9ZJ=zH^>{Mw?nVd-U+awc5)E&w*YK9$hT_`uurZ*9f{&vXz}#!VCWa"
        "t&(fD<~29_yifZfSru-eD7fIJD3Sq0W4+1#oI{EXD@!Ijnhk$^ah@H$aByCnKhBP@-qe!W^F_W0$0^L8!nG6_6_uI=Rg&gC^>oZV"
        "A2TMr@y5iu=T3(G_r#{Z0onB5hcw#f4qP%Sh*Z2}^+e@;uTE(m7!}UK@)Y^YgK;t@oego*ap2b+6(b$G$o?XOIuc#V7VitR-M)ny"
        "e_%aLm-9K!q0h&k!@uPMnX-*`=;Ts1YpRUZrkAc$vE>09WOk%ZOJJRSXpffk!J@bstU1%GSFF}Jy+Y0Qs3=>i2{3kjP0JBdt5;_@"
        "zxeI`K`YKP2qy`fX{;5}-cP2gzepvLiJy%(q-m+*Uboq#<HKzI9NT>%4z}6j=8w-K5s-}S|E4q4;5z)OLg)3>*KS%UY$c5{srFPE"
        "+mrGef#q=(CsT8J6(8JA&GD~`HuR3sB3Uz2Qjci-UeqY!IeF~PU<Sa@n7(iS1v<cfM7pn-k`CjP`Z!9-Aci6A(Fk#m=c7yyi$eob"
        "p^j-d8PYbA3EzYy0N3spJQ->Bp&w_{+VI|R6)Nwx6Tg|kBAX7aIh>_qponAtdG9x=C*yb{x^17w)gaL10G9s6a1dhBpGEe{m2D)u"
        "k=-v)O9KQH000080000X0P5!>`^W+S03`<i03`qb0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJ*0FWn^S&ZfA2YaCwzfQE$^Q5Pq-J"
        "|8V4m7PZ;LW6=<3R03&3sOxw^sB-P=VsPwmc2=@q&pAocrE53QmsGy<ci;Ej`I02LK7Ts{NzVJJba~+{+5lTe3PPpr-8=YpH3uI="
        "<-ubiNCZsb1`(8#{_Nx8^7EJF{Bn7_n5Vf)lO!1qhJy{c9Fzz`8sR;9(2PuBe-xA2v*1W(skI<HSQ`Vw!G|t!#83}tFP2o|csQuf"
        "V4km0DQ5T087->?`qG5iZ~*)Z8r2G|bP8#uu>Q_q?VM@iUfe%Qh=OtqNPY&|hBhS=#5vxTByq;#sbQsX1EMHQwTk|%aK~a*)HFuT"
        "?G}2*V$zQ2q=|KWPRCXzUu7s%opYrv3ThMHOge?f1#)~0)A!HX!wp($gibfO^)Hmc!x}e^5TvuLx0HdeL&y+Z5p=E}kkV*Dg!t6d"
        "1nuhtVP)nhg=NME%5k%;3z1|!IgVoQ4d!vIWQOv0;n-*pMhoBI5qh2h5wx=@Tnb|tJ1>2JjWfn=Vg9#3(h=aMK>3>Iu*#V`X*qS)"
        "RM5DecKVt>dUNP#x<!4Lg_PS{AiTQR#e3(iO*Jo}J;HZkN}Llp8YMgBSJw~$8<rUl6#Te%QmjiOsG96>VXIZhenO3#CgYR5)@(&|"
        "Gy3+GAK<eb7Pvv;_G6#L8X>|=H-x=f$A)DO&k1etf6X~zkR~3cN9%)|`={hX5BwDd_BQd{D9E%@Xd9zC)-AD7zt{m!h4)$>BiaP|"
        "cX<d8uZQZW%=&zPdHkZ{n|fD3w+npJj*`>j<M8~-pye6`*Z4i9^*(}&LxQ%<(Tv7VJG2{=K5;<970>su2bT_owyTOS&7M*@e`Y{p"
        "`99Q(7TwyoliO#?10hkzI)XR8As*nWUAVhSXS?^)Eve41<1hMzI>JzrJ-!V`N8ZJxF<=78u8a<`@IO#X0|XQR000O8001EXv;=<2"
        "5(5AL0tf&ADF6Tfb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVX>MtBX<=+>dSP^FZ*DGdd8JiLZ`&{oz6Z#E;CxDf)ahZnWauzpNQa_p"
        "k#$9{h9c34Hd!(#rCo;o_)&f*OS%H{$snnZua9JsB)1oD=O8yIf0Rz!042F6w8BaRVHM=tuq6a5f__4HM!nty2qdh}k<uhd#-s5l"
        "Cs%+HK}aLKM-RFv9fe*LjT%{SBsZzGAUs&xy<X|uNnzGl6e5+*=6c<tEPj?qx<DJgl_-zddnxkV8P&eHLJ_0BbKxF^+AWA2MNi|="
        "$1bD^!~8;f`9NDd8;>d(Tw5KqF!~p#>+I49J0FjLKkQ<dgAddQt#k@0TRFAMyxWq*N(;Z?&RN_ydUmAzKdnlGCv6ngHUoXF@Ebv>"
        "Di0YqZsX_$2jsq^jM)tUL<pQ2%d#M7jctzm1^cT;s6IKax8<k4DHHSIYL765Q(MD?D9zT#A{s6AkYiKeUi)o<3RXMFg=Bi{;P&EO"
        "ORYidbLLBWTLB)c%RqfP<>Lr_Jo6(Is0*PmXHQ;|USX#9j3%CWIX~d(8JvH3flqeWOSpsxkv7tltA-}i%;$Vh0cbp)WJHV7NIR6K"
        "$9`877)Z+lZ=iV)U8rEIk>3l&bD!bO0df@9sD#{YdnC_Et*_`ApITAiw3qGYKTsT`)D|FU%-T1tNi|V*NlwDO($9Yf5Q{eeNw}by"
        "C?nm)%3#uK4m--l@B3Lr9N{L3#pleRAj^Z@6F7<Q2c4{p6RqO(hFx+tJ*av=*pUp4_N*s#$75tC3@LQ<C%dtJ1+!n;0`q?4P2AQq"
        "SB7(K+cRREP};7i?XvhfyO}*!6l7pN4+U3{^=Z2YmV_RO4L`@N&)oA`Z5h=%DrWa?xtJ}#EM_;4=W!wO!?;^GCmDPS56#JH`wQ0("
        "-Rw8T<+EjjgRnu!wG{>m<Ymag@WIl7ZQ@>2n)uJ`-t){rK*5)jqMs)u&lqV9de2TMreS~^X?BA`D<}o=Fa&;3k6X39A(CmZh|;c<"
        "&7cX*tv+_xCh>$PQ%LUWDh_UWd#7O|>KR`lTHsK1u2H24GvN<VO9KQH000080000X0LG+}9mfU$0E-p?04o3h0CRF<FJftCZ)0I>"
        "XJu_|VJ~fOWMyn~FKuOVXJuqxVRUq5ZggpHZZ2?n%~<Vk<1`rmK575MLlG!}G}}&q(4a^Y3^?vK-ECuErm1ohcQsAy)ONb-3Guh{"
        "*iITdO|N$nAE-2~iJ#Zs^XiU9qpQ=ibKrGJCCf+w`2UIKW1=<Xnh6e)&Z(q4qX}@KfxvIHED023LnR(o1<7cMwCVBD@zEcGaSck$"
        "bxGa<;W?-Uku--_C-+cE@fXdEPzCrBSXmX8mF8(9^+vq8OyJGy%jEco6V28XUfqw4491^MGLkK*vSRT|wydl+E@Vjx_MTXCOHx%x"
        "i<Xsyakx7D=H%-1Tm1VD4ZNika*?T%i<_T+e|dNL^WC2}muZ>9Tu9Ih3iKVxv@tZVt)ZX^!lA9of|X2LB??82MkC`e`XHnREo8O`"
        "pHj|&DB#?X9UtX%4zrpS`SdSi^>ob?o3VoFO>}$&_(6z8PUUoszU1ePj62Gekcp$n=^+Y5i~01K@mxHggTdx+xE7qk#QsB~7{)&Z"
        "jOUIKy?I49E-Vj?o}@MD@rPD5WTaN4IB_ayz9oUDLd-}}Y{2?e(A45%M6-G5TFP@)!e<kB0SJ2FQ+L#uWj>NHMd6+lHNBKl$Y^wL"
        "Xc0J!wOT4DYo%d9FjXFAL}!ZxVOZ5BV;@cSRcK4R#}s?~aQbL;hL*G^A2j5$ao%J{i;y(|6nl?}nzK39j<4Pe5HMv6$_B_*ig)(o"
        "1WrGw`WA`3B5l|#6f+8yP|Ot1R&_!7lU~5EmuHWT$Z?tKoHRK?B;ijqa@+|ruUSSXp$zx3(bHu$%Gk<#u-OOakH)X=?RjFIe>{Ya"
        "2ZSg8%RvP{4akSFFWIY<CRfqaPVR$a`Lc)9Oo$?iTaiYUlaooIlqV?6uU)Q`zN;|U(fBNZ7S{1ugPx?imK>taUl)dq=qU0nPC~sg"
        "hK7|Bl+(l1d=7dNaKx@#Wd%_RZrav=WrxB=n{E;3=@qs})ZF!tTWrEbH!WaRh-`_8iD~!S+Y8r0DUlq90a2T>q#6efELfgX?u~}D"
        "X{@f{&0#uaoayP*1H~YbbdN3M2!U;S-cjrMz!6To>4K76xkfpG?|X;Kb$EW}otuNL&>ru>9&BOua$40Dl~J5-J&n6FEQ*rx$a69A"
        "ok}`4lECO>Yhv8*CByw>8(Wh}I2mjk*xODT)3(atix9Fy+TPYwZosj&N^;g#(QJc5-<<{;4YH{Rg;#F))T8rI^4i`mUFXB3^Jjpj"
        "^{pXmAA6K7>PXdo?NRtWZnE2#cKqxr1IeQhSS-Vrxd|IC5pj6rjXG=29tg~b78o5A^_p?4&ky9<j1e{Yl_0sNtRJ71wb-Dvl{lQR"
        "@YNQgJ)*6=<wwZd#^&QhuK#<HqSm`Yhh04kX@jHLYn{BCtJ`aQh&my(Qj;?}{%#koPvrU>{=}vONNacCwJB2U9=2_B0UB$U6H}B5"
        "F9*1?<3_x|(Y$T*X+OO!8FtI`)bHXb`<cjme-m#}QMh4KHMrc`T;-ocwtrLhb?&Fyw=SDcvumP|DGu@kre2$I=DGDM=$O3%4&4dX"
        "Q7b;A_SWTqVSHve`fYEAc~G0rJlIX+gWder$aAYU?!3b8ZKYj<$-q?U4-sACEif9`%VT3#cp6xK{^Bt<ZaI?B$bmK2^QaR%h^G;J"
        "(CMK|h;!Q9BXOoz1;bUx3T`&N9oL)OG2Qv_g#M+uV6>MTr!o^<n|m(NX1l@{X4TMQNqL~mkkNM<Ci{j<4`PoMx4P!#_F7=~(Ga?%"
        "RnhFCXP$)B?&R`DMNd#~j-_ixpQC=Qg_Ao8XK!Bu_Ast+g`pAd+%JYl*OG2rBs_O}w<Alo#}hZCqK@}NUNpLr-QD6paAdv5!u9{g"
        "!d07uZo);YIK<bRW6`qj**cvU0<$m1k?L|$s+H8;`&6xJA}M|;DdF~-wWPE%DS0Pr>?vP^oq%zw1#;zjQ~9I!Ns78strgt>KRMaO"
        "A~buMN?K8(af#PdVlA#|<fQEugmGp43G^5>^J@!o?fC^M=$=W~{gR!%(_b1gA-%%J23FQZv}(~r@V3Bqu{EHMU&JN`-OL}5>As|<"
        "LDGCWXGIbH)3}m=6FV9+J|D%wLUOSdEQf1y-EIwVas3O<!)$3E=5~^4CTZcCF9K5$yhY})veybJo>jWn#peYsE>UAfO890sN0Npo"
        "X;|J<ylhVkwxp3e@a@I2nh_fY<~!Z|1d77rJpi2UNkKa^a~tg-D1o8`*265l+HX(xSEj97)4SPIEo#-jM|@2GZS!EDD|la<A2)(^"
        "Icn`gx_<#sO9KQH000080000X03MT~bg=>e0D}eq03ZMW0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FKusRWo#~Rd7V|;Zqq;zeUFrX"
        "82KrYjHwV8DF_}aT8h*ZL=ik7WUakpXT*Bf-Cege;^Ue1C3VuY0xZSxWiDsVnHfh>v_5*Vg4!C{<erR%Lge1q7MwCZfs@->QBEGT"
        "@N|vfNw+{%t*}DvMdv{d)<>_N_7+LqCQ%g4XY(1G8S8<g<(u8RL~9TZv<_y4H5Fu3m|Un5t4buf(FK(-crXOC6w+PRm_K0eEyH%B"
        "%i2g(9~)mcK88!Ydd9zF;RB5O-YyqKVU!%5Pf_^B;)U^_QONehT4Mvw`7Bq$IXLc}-v{R|(L(ElFB*l*)qDngNGu>tNs~|0g+o=u"
        ";28UAua?m~9IoLvi)FHPTTb3_gd_xjH(i{7WlV!QXAEhe0B`+}j5JrCl60$q?=PsGu`vkmIX8=zR>J+<eY=DwZ+9L{(6OSYO<-m0"
        "gEP(x-;d*S{*T}B>Bno(2Fk#r7q&zX#)53Ms3;#mTF6Dd!M;7l$X1ZUgPboyYeq3d--z@2%{F&NlAm<lD%2P|iI!7ZBzCkYQ<8U!"
        "T`J?d+lPp{v*-ia1AI-41`Mg=iNL*B@Eb0S#sk*(L+;xCogj_wSd~yA(=JEAoj80Qe`RGz=e8eM)oTD5Dx*us;dypn{+}d+97Eo1"
        "YsK*C`~-xSaK45O`|e3o7<(hE99lICW4YQH(ys2w2p=kY^=LDM078n|Gp@66v&RiT*<$<0%1;}mpqu?|G$mR--;gC;hGpnHv5m#6"
        "hLMfc&u5|btRW@S+9K~O(kRoaTObFQLqK_N*9=_T$v>>?gc}FKZkq&WsTl`%)rO=*4bt3bWrPI0CCAHhgLX2EkJLWle|_{ku<&90"
        "0(3fGGwh-r6$$%`grZ;l*C9g|oVz5T=cVtk$`pShh8CaC{s2%*0|XQR000O8001EXZAJ{~01p5Fq%!~jApigXb8=%ZVrgb?V_|G("
        "Wo>O?FKusRWo&aVZeet3c4aPbd8HfMZrnEZJwX0}oqdr4S+(7?O<kziZ8u#knzTU^^u;g)jl`Kzty@IN_O$4J`<`=1N+d;%-F5(H"
        "Mk0B5?l&@<&F05!DDOD?53lQty<|h#99h{6eBZ-B+g!0eYl^mJ_q^O61`#oy?Rn3#Vc-o;h*;CIf!AHzXBEq{{J`Vq&z?W~aNun0"
        "#L)AshVh4ii6P5xS$p3I_U_H=i1l2Q#h6t9^X;2=47OoEx9?uZ?5CzWvV!MrVQ%#NSt(#mS7kXbV)j0(`Pr`R@3X#O;*j;cU_bl<"
        "sOQyKaKX0iaA1O0I|7{dt$=qx3*aXJ!OdoaSeA(Qz<w17f7b(@@=aB4<C+gykqw!8#qM}7O3yr?qaFJk_EZyhWyOV>HXOUM+3S%!"
        "6N9ns(DwPkeu|rhWr8(&n*mT3KxPe_YbYeEmmh$q-?h!I+(*Wnx4^W@vo@}AK*W2RDW3qbR6c9S!SiPfJ~Oy~O=5OO{w?M6COh(8"
        "4Q5?e9n-!Yvi_KMtw?#jRYM>*o(<{k9XyA~0YY$F?%F)72!}W)S~MoFA3ENbHEjOA<2h&uBK*Gmjjt>PTJvGz5Y7H+9J+Cc*n9r*"
        "68^r;hWrq*5A7{)08a5U@3OLg{gBmN<%No}2hdfOfn+pfx#~Oh*0)SC8koN4MLMak2CyB=sz`rr!N$_NQk2`WDu<&F5Rg<hAfve5"
        "{>pQ-(uuwfX-P?`Qvlz$-4U7k{8_<wtRag|B|ifJv|i2Zi`a{ZjQk2Lq^(%OKJ+8J<qe5*diy=tB@eHp^uSoYAFGNLC=j#(!NA@a"
        "s1`#b0izQgVhB{WV7p(l9SEcCk1VV1v!h_60Mv2Qn9L4AmOE`v!N%0nO~^h;*yi*2;;Z<4qxL5a%1Xi6&m%Ci=CAv{?E~$*40pxA"
        "N8Dvq39<u!knMqkVz3M$NhIue6DqA0f)7A3S#7YfL_lMJQ;hNgrXPacMp{hzJ2Ez|+I!vy;U6)25DgI!7=hq~+zL#|3K0+_a1~jR"
        "qB9JL-yqy%GY-46SDP@r-r#@|+va9s6A5eP!+lwRV@ud)mpF(>uml+P`tsWi1HI$y<*RwJ#3UX&<VP^Sr!gRNn<9YZ;__9rHkg7P"
        "rWIOw3M?20&*Vc{6ugn32t8*U<paDgN<c!+ddXfe-%JD2V&9kUT;wHaWa6T;ui9-^rPKJu7sdojM4B<6pD9JIgw#Glg-y3*QT9@+"
        "0or88$tJ6+4f0-@jO6-{ZNn98<;aqSpN`SS0i}Eh<o?!z824z>65SE{J*m$iS_-%u?MH{PZ`2BD@_7Zi#iS{`C8skv?ZD&64V!)k"
        "k&E|WK+<D?SLq?Bf~??04GN5b6GAFPY_OZJanl^Q1(P|;b~uf~E5&RTP7yG=AQhJljl-#i7_^{)N-m7|5H(7~5<>?s&_K-5NeF43"
        "iCbep#(xMe9*7Wf(_4N_CCd{Olrj9m_^kw*TFAKz^d#~!?x0Fq{DUuE+yYZ3f<nWnL7N;r(UCRnKzem$>$miTzu+|64^r+X<Ic<~"
        "R<vA#5@RRx3_DrVUQ6NOWddS@m@x4876Z%yi+DZo81D8kDdt<3f}VdI!4w7K4_#H}<uI9^!VVRcc7;gFP&Gunu>l)2Ze<=&@n#SH"
        "hoxX*7OgQTQ^zgc>cGdcz*%^95s_BM+R+$$<Sit%qD1rcX!Jbq<H#XOCKAraw1E&6sNY@R{u)?_h4Ed<?*qao43|KB=~qC5(S#7V"
        "wno4^4h|N5<7F6AaD?eR0D>-UWguKiL53wwj6AX<RM6T;dQ>FEF~w_wAssm+!~DiDC2*V$C=8RJqO|6YcwidBVq49+TZ47Ce}Z*0"
        "7%%gPu)o4O{7ejyw9E+JJ&NFIz%uQd)N*M-Qc;)hy{0YI{rcj@vw+Z5(P<9H@`WkavUs?Oo3Wn#7e!yO22xv0v}&X?ou{SjB}p;U"
        ";FBwJIrNx>{vvRRv8?7GU<OK!G&Qbp!-z?B@~e*lhh9Y{@=t)Jtylmcp6(!xPo)O~6LpO5QPV|rn~mMr1X3bMgbNt=mp6eGG$K1x"
        "C^st5e?vujUp7U1ZvdD4oGM6(0WGmrZr`_KCya%fh4WIy!KW-1dD~MGF9lF%Lp@g7o!Vl=NN@(shon|g0`&t>;(;Z9#2BK0(NJ+z"
        "!T%KRBDRA{VF)&7WwYCa>zLdhc+Ug1D$bfC?SIR8hoZ-74zxh7iPVzcIH*iomA5>Q115M-^=(FCDoq?kyBq54fq)k%{Y1hdEI&i%"
        "*Gg{0G<pJ{^}Is+`HA&S6Ys`GCE{@b#cn$ej@z@=M663BGf5nC)~MEEvT`1u1DkY-ozO)6w5S&F13VcktP=wpzke!Ya7~e#66Jnu"
        "M-hbBBSKfP%pt5DD7Pn6TIqr0rhur7er~$|0TMo{>;vEt{1_{<WY<~Y)O&**o>IgucGq?pKnA6A>Uyl-p)`XMb4s?_ZUCKhQ~VUl"
        "vT{gM;xQ;($MdU2*T&atF!NqWNwIv6Ct#-9lPLih)JgbqMs$BLahqR()Bk{_dU7nN1A-_e_)q9=r}N51Sbf4CYOTA+)LN0I>3Dhc"
        "&c<SUa4Ial3aJ|bCO34jHtJ|Ug`{4pWs^92=!l`2v0jP5jMPFy^WYR3r%~~wGoai5Ezp7?j)ba2(c3194a6##_@-=l)(6v!NVOT*"
        "w#<aA?i~p@kJ!_K0K^}_NHOri1MD|KRfUZLi`O(R!mB05a%Dp8-e9F=Dkp`@#dF=0tW@<LiANIYQ{bo(B_q4+K@b8l0;A4DYz$!p"
        "8G867m3D=~c`YW;bD!Rla#d3<DE+t+kCKT`bLYm>ip*$H3a&T3Z~H<7q2-k3_c+gTH3l-VJj`&7hbpvmzy`5UQ=ViMRDVwHDn=Z%"
        "ZtFycaB|Zv(_%b(KM6g9b}C_%JKi;TzLJl<;K{1fl?%+N>iaai^^g&-OgvmQQ}JPXVX0Px6<@h;606@Vp{sbS(Otk@LX=+<O!@Vv"
        "h>^Q$a{h^XlzGjELt7Y})-}BdFq^TlCNT6KFmCpMcs|x*tRayK*%`Zl=PIkav@V<A+)`et3K&zULoE45y-z5xAgQstUrzhT)lJ=$"
        "S@X0{Y|m_z&pM;LV4Dd%HR9}mM^vqjY;Fr_!g;DK(NUkfdl~7@2^EhUeh<ZX2hM-#hTNUbZy=TQJC@a0WZ*96IKj->cG@zTMdS1^"
        "EIB2sO-uYt-5~@h)pJ%ZN8rv<w`px82@smZV3n=&;6Fg_tjnkqrTb?b8}#glPk)Kj`6-T3stSXY4Bj8ws*=4mDD#wx#{S%Oscov)"
        "5jx4gMP7_4Vh~=(aV*o_8(9_1_8|qDR_(qV#CmJlCl>979SLYmPj`AriYMB}#B9>p6s^P(hiRp<`wQCj`yYQL*)n>wc2m}qfH)H$"
        "4<sk8%;M_bury-GI<m-TK-H8Lp+(he5O^{0<kJ>7Pv4s0%I?`GiNh%c`s(t=!_=j7e655=Th*f&;1?K;)1J26qooKG$DY`yOKeO}"
        "wD;uUgGsJuvVBsqVjqsc!1w!fa7u$5CjY4;41b)-vj!nRTEelEJ;7Jobt`)QuIoO*mXnU#T=#UhjUVTXhy-xTn}a|c_v3JS;*0!o"
        "eT7YjE29Z7-Kl1rN|3d~Hcm9MZl%`Tm3Tf%U{Y)v!OYHTq5zJ#dR-eP6l8~K$eKf--hGNov9ShUi)bU~ctS*j2k9zKEYc`TsNN8;"
        "^!%O_&tSzeBw5;adP{q^Xc0{?OSbC&nOc3yDoLbxeP!KtfrRrL^DfNeDdRHc5ZEcEds4ekmZ=@FX<_2uYO9{k^D?NdG3LyM4I}WI"
        "FI=Y7$B3yIA;o03Ej*`5zI<-VuU80*9EzPJ%-%qub85w!rJYh82~TUy0yRD0WR@0_R(hDEx`sw5JBxfbil#BSYn5$*#bG#KG?Kl3"
        ")e6I2S4wxE0$(qXRMF{LDrqUK+?FOnF*fVRMsFqR7CYJAj$^^^%A8+eMTmDN@^0MF));Ele!8LGjNrEf=S6VwrM`(V7TM&7B&a}s"
        "g)w%y>8cG*nr|`Tlc9F%xCUNg&zBEr$FtkC=5^a2$>N0uDAy3jrgIZNFR9mVtx`IS%5)uEM(hIrK7+rPVZ`Xg=i!apT$D|+Yi#ZW"
        "p^5+`T7Y6-YYRd{A+K%S^xvNgHsaU}2M*Qp*~KQB5Tq=P5IX-nVqfCc7q8&yD}3`s=my8ffnS=yFHGRCi1hO%U`6VZi!GS|P%l|c"
        "yf7|>^@ExLg}e~Ub7*5Q-e&nN`C_$xx=Mp0s!A~gv`0L@4|M9f@3W!={Uu>`F+NfkvcuM+0SE7h$Wa3mMcWSB=XlhWL19IOl+?pg"
        "$dO<l#yAv)1bUC4r+pP3P1|6mO&yn70@=se7EdqmMaaH^bbEa{pod5~`%XMUPoi+ZJQ1fVq}7w!{%JS<$OGV^Pci0DkCu(RbD}&f"
        "qNuT(B8l=cPRh+E_mPmP;Ke6Hbx>%@#<WU#8+v2amZ(yR{PL8K)a*>8_}Ba6$Uke&-2L~|u_ny*>dS+S>{%xjZvq=5)y!T&<J^P8"
        "0+iDiH!Eb4i##n)$QmrYkhz4pU{?DCb+5z2T8^sYF${Cv<R?<J%EYu#P&H6uWA3CPsZ_*7etEG-u?dnY6;1WZM7{7WNw-JLCvEIs"
        "fCY1TyN?9)I6-<V1Ki#KxA%c_%+tgUV-d!t6c1kUGGCzp8)2)?9(25#W@*2v1D;noXXxr);c4J{(vx6gzt*@xrEVi7PTpi|?&`%M"
        "Y}eb`@9Vz9a$B!Y9uI^*3z5Z?Mpi;|jv$`0i^*efV|YppQ`CRRNU<*t(A{cs3$z{TtiMh91JB37nX;C~t!Hr4X|8JAE5Ao6mj8;t"
        "d=&j1RK-I-<^z*=YN5u*n~7WiJi6!lHBx3>Mg`mI@PfF&92aoh?eKpwO!voO`d!(MbJe(qTuYs5r-Y0S#Y2Bw<Y!_r*}cR-T~A*g"
        "=<j0tl2`Mfc?#2DQcpoglLrg`4^T@31QY-O00;m803iUYS>d*S2LJ#~7XSby0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!Mga%FUK"
        "VQFq<WG--dy;xh1+%yz^uhjow%u7<HX}gG06@s*su2iX%3PB}QRgsf8GiH-G*v@nZ#DC`;UlL~~J6j~guCzN7``piWE}@jZyct?i"
        "2}62j#GaFC$E)i>ir$b`vRd?8!up!9nhl1lhq7-EMDz_;ystP>TpOu?QuZ08^!~m3_e3{HWyp@{T`S7z`|n!z_wO}Ic4S}%kBC2c"
        "3tuBb;8*SZP0bo^c(0|>nSCB;P_?2FEk{CowG@r4Si9w2$Ffe=yw%xkOJNl2HE10egztsnH>Q{`p5f*l<QbHsXzRi~E9{LmQV=70"
        "@L|^RPO3vOC|OP{oMlCMFkGJ{9wdb55XdUoH)1;~UbyE$I{wzYalUDh%|gB8{yhRedf?T@eXMx}2P(4OA3$i24)Qm$=l=A;M7|dl"
        "&uKN*jD~0CF!0=lDk~e%#5{TGkBsOK`4ww*;IrebR#JAwR+Ra(?6H5WIj{34U&FWbxqAYMbv3pvadCj0`(qS}lNZ1KaQSFZylKUD"
        "XTBl5B(?n5qxqUhMP9u9l|YnhaBIWbwq(_{%h7e)Np3SSIXy{l^GYB&JxPgZ)(C_KH4Q;;60n8aPmVnh;L)l0OlX_xNs3naFosZO"
        "UWf(hiXte{N&`}4e|lqP@9hk3#~;W*Q1S|n-$-*M$G(24lvD}*UYcEqL``hA5rh)aa#U4ZMl|V04T5ikHW^J9#-3Pk-eo6>y<_dj"
        "gGNmAR)D2e9FizraYQWZ1K&ukSd};v$+zV3Svha`))j(4b!-$Sdc*;qIy^YMB<^3DAyUX^IdS+2@o`u25ypk>nP^#w!HGDo>t!Lb"
        "QoVhUw8e`>$w!WW(F8yNaWlN0Eb#i)LANE4*L5tLCr=AK4p^Ib{i%wgb{LOQK_fBHQzkVhR^<CDkcnDP`jx<CP!4&g1vs&_%@rE_"
        "vzba%#|@MyU!Ck<&+xW3kW7*?56I10xg=BhDX=l-zo?PV$q;9E<Q1khS8GU&Q+>XEtBOlM&kB3So8|ebxT6dDS`5WUscM~D#f{)a"
        "APT12<0HBi7eqI_H(biORtK?*J747Lfhh)|;!01&2*Ns>b`EBZ;(S92lE>)@UANeBQ^3F?1Mn7^j#Pm7g?e8Q^zNc2JRlbj9(?)u"
        "&fH!e;}3n+>#=EI!h$q+;@<sy`6Fo%ORkOtI1cTor=ntk>H%tpt0n7@zSh)^^&>QUY@7-H@rM=@{|+eb-x^RlHVjxza`G4T*+8G)"
        "o|{(8Iq0N+|Hm-#3cjbFfPTPK_%!S|ZU0By#1YC0q_8&<|6Fl1s@@%M$X_!Z<u?OYqJvxl?^|&_AZBxPqk`&LxJInQJtK=+R3=H|"
        "%id{l$A5|seO?x}RM&9l2-?(1C`1kF2y-a*pjspB)p*WZy$b>4aF4#WlLO8l8Wi2Jfj_$D2eN0asBPxMOvhckVRi<zjDt3cvksiJ"
        "7aA`Fo*9h=VqRm0VtPl9-@blXfEcx|<*0d4F$ne1;onEowQfw|HzF9Gpps1+QR0IZtsb>dDS7xELHExS?K@|opOp02ZD306M0<FI"
        ";c~+eIP~CaB`G<`!P~6gA`l(A=ZZOxwopE%$@EB#fMs<Jy*IZzS!U<uB(YSF2>tox<qyAQctX<*99>)NxXBXa=5#d8!)G+qXIIBM"
        "*lBu_(BP9STHK)TLZ0=afmPA-jZlueAZX&HBd7VS7F(c!Q@=}Q5V(u4o+e1+RC5g^O|u=pakrCYeRrm2&;t_Wo#IgAF5Wq&Tsm9g"
        "kt&3WbV7H`K<TBEOPztYIQ7W>Pw=bN-4mhq6dVmAskMN{YodixDdn}z%KZgBJH5V|R(C&vO><Xd=l)_ctdbqLTa=;&gn@NIsrPho"
        "LO)DuFP=NyqpD@~cXe+nUuP8#VVHT$6~V^9a|;UlT{&OVBP`T@agOHpuwm=F_ly^_pCB)>P4n;yCGhv`CY?SfKjCW=iw?qY3eY=J"
        "!oC7s9zjkt)IUE30|O(VHvH-K7TlWbI~eDNY|@zZhlE0&DOOxkM4v@rfFn*q!gmns<GNT3S7$AJ4Ve+W;|ZW9P-|o5j~pMyn0$UY"
        "lR&ASFtA!1(s@R&hH+vgpeQVsCO5&yYTvO6tTp&(>RHSunR}y>?EzC|q0#kuPM%ymJ6a%4`nLnf+S{0UsNMKv2rm(Qd&WV^<~}jw"
        "0G=8$+x*fff&)fy9*$3=XcIRJ04}zQ)zMVt*0CVZ;cih;Y_{TZSs-hO1@3nmPJARS34Wh=d=gM64Xqjav{w4$b#{AM@A2-3-~*J4"
        "pCV_t@1ma^m<hwXJ5%{oDaU9{DXb_kaVJnq+r(v`&<)~|CoW&p49`;+B;S`oC2vEvN5OZh^ac|KnFe4@^z`<f=aww<!^R@R<_saZ"
        "ojH<QtH24`dTFz0vBb|YIJMZv;Ha0*GJIGoIUEY_Ll{JplM8+|yJIZ&V>-C`Xi$zPWZx&BN1e()`Vdcg+g6siGZ{WC?#9m-1joJO"
        "x%c`!A^rnUO9KQH000080000X02N1gdYc6R03;9q03-ka0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FLGsOWpZV1V`VOId6idNZ`(E$"
        "eh-lUK<G)@6g5t=HZ$M>>);Lrns!Kny%d8$OSH{JA~ljq;uYI(-#MfxQHq*$1`J3%7oN+<!%tBZU7emUiQ(cv6cwecPaYWG?G2$?"
        "vwBk<$W|(1m<E*{X*esGPKl79Kz?O)P05B9cN-~K7DdtV(eV+l8>tL2QWg90TP6g!B|@OuR>_)d_*NFQ+M%PYkYdYsq_g<#`qR7j"
        "AJ#W1SwA*R@tO&9)3D+zmPXOSXm5NaOICeqP1Bl`+_1mS;QyAIVxN**dB+6*o0ZqBp<JyWY28$8&TB~xx~?h;(R<I%>X&YH&aB40"
        "bq2*_WeG*=gQ87S9r6dM%DflmjjVWac*al7)Q*)oH3ka6r5FO)GQ~u}uB^h(UOwjvio*r$mdVYZtT6nZ^+7dQmAp_=>l_>?*&MJt"
        "$45m42q3orjA18?C>A1vhqhwL^7x3ruVh=2=dgjv^H{TLn-UF;T2305l6#0~lWQpcl5k<jiY(49y%+rH77XGf>%Ap|jBl;i3<7~L"
        "pdXri1k?rA_=7|>BfrzCWoxCRiX+Tm05j}>X=`Hk43OO_()l|nne%&36{o`JEDAQqtTNmlV%8RfUh=?8v)8cvC0d`ojh5v6EG5zU"
        "0v|6f{}~j!ee#mmpzX{)hZ$v9m_G)CE}$8MPBJyHLHcbFW*8Q9Ywf`n4-H7ecv3)tLnALASEtjweIPHo3?hEPWmV^UMoZ`H0~d^{"
        "cw`;A-EgW`?`Wmj)Xnr5fxl-CSv-cy({iOV3Zvxb6uML<BEwnUSJ=5&+1||vp7>6lqNbt;z@{zNEzD^RLm39c+DSj4aFX4qy%*3M"
        "UgulJN-%t&s$5-=@5VYM9DUiE)oFGLCAU*y%at~(Th+4J*unySEwp%)KuQ(}Ji4h^F5pUmxdu~9OEur#C!=rm)v5PP_G3wivav+8"
        "u9JFoIU~8d?}YwdnmLM@SDM)48g4jpJWW>G+bS1OJy*=@tDBGT5}u0$2eZp!OS(>W6W4ZZIlAi-a`IDn6E3H)!t)n4@x=oY+g<uK"
        "E-+$V>STH_RFU7YL+)olP4(UC-?4GuRV$q@j89z~3Jk<YJ32O9NPcYyY4Boq3bLKTDxOD(+ccd6*7UT0V`hV@-{V_~P{|!Pee&J7"
        "iKm!}C$;*%Pdf>GDP~$FLQurceL6=3dYKElq8lJ2N#ghcLQu6LSN=rix&~?$&tLr|yXOU5``X~Bh>E6-EIdnIcw7Q~{3ICB2LX4p"
        ";e`kID<qiskisQs_P{+!19>2&fu3%8g@hEj8UdE33#piPR1}>$-dR)8(kj<5|Ki1)t{#?OK`=8E2Xja=BV!kKXTTg)#8!MBN;=Oc"
        "<EYQ$WD%t#23D}Z|2h0Gl5XMjhm^d*m&GN#{fH~ClkhOT?EK6-2Um;Nq2o*YayoX}(FNO3J5ED38z-^P@oDxBYhYvR=8jz_LUsjI"
        "(rIKBAPCtRej^}53(c)C^9}X%uAz$7%rJ$et++O^-6I>;V-Mn;q9q4Z*(UVz443VPO1}gIGxQ7WaiSeW*cAnIaAhN96+7H;y+tC*"
        ";vo7YL7dqSpsET+$DEmuHpx&IEHOmCh!OVbI6om5c1-rVh4D~(F5APD6taeZsJ#^%1$4ZCvmo#T3O6Pg=&Rm&Pi4E~|7F-EV*Nn{"
        "!fLNH0aXeh7-26!HehZz@=>Vg&;pw7Ar!eelIuNfP{FpiFDZ&LD+-YxY=_B!Rfr(Sv5rUTW5#FqQmh0s2c^+9mUW-F00Ac*-6)-O"
        "oLWa>-0+<`@0S0!TRX!n85)p^U1;lg04tCWIBv;v=pGefn2ovHQ9}V~X~VTFD^%V<UG_CqcR71xMQfgD!3WHLP)h>@6aWAK2mk;8"
        "ApnCIu8%zi006rX001HY0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!mlb#!TNXD)Dgl~>zt+eQ$5=T{8Whe|Ci-x4EG;v#XIv_&or"
        "?9?xTK`+UXv=GT<cbB%6qW|8RC6_nJNo&Bi#GRe%nKQ%5Wb(5rEIwE$g|^a4RmfrmqDbi%WR_LNYGLIaLW+q>QG>y3dGPic&{&xZ"
        "t28X>_d+gJBC=mG&&50lKCRIpR90rB6Ah#a3S}Wvf)s<3f><P*SQQ9#UFKzmITn^qHI~XqQoMykBx{@pS8P&POXuXEw@hmBA<3!~"
        "0gJ{r%a21yv@)i<5xRhK63O5m<!WusJeW)-L9o;+hg4XRWWpHINRB9NyGONz-Ikn|=I|Gp*bx3GS+&SQ_*9YeLC~C%<H<S*f|s2R"
        "_ZKczWih>^=q`j!){6k>hrFOCnzB_`GYI}5!#nafW?3K!DiEeF*TQV`9Icj#5B$Iv2Olu0*sr+@k1aGJku1u}Mlv;v+Qs=NER50v"
        "nKollF3e_+U}NX2%Dl`sm9<uljVQxN%6w8k_l0lZ96k_Gb}pq@+VwYY1wQ|j+6L{e6uPazM~{62rYg%!QlhQeU;);tGA`|<R#Y20"
        "`VUEe$w-t5RcWo#CM!#VbCy&YK{WC?;%16Acaafy1TZsoqWnX~^3p~b77LKn7Uyt$cKYl|*b1z$=$}44KIzViDvvU;MQ!}?>FAV9"
        "u2H1z`LphrpGT?8{n5$Ov+gk7Qh!Ayc}{Q?olGq;gM*)3cYG$+2w_R>Zf_Rt<q5M*&Nv>DN!Qs#E2_x0)UU==Ik#L@5V7bSmQ|MZ"
        "1ncPLL96?9%><^Ug0)OjD&QlG0B;YkL(c_dE+L8&4gp=Pjs&xeXpKou-J#~x%xm`g-oSbrYe^!lPQE}9n6=Q3L@7=py;h&&A0!1t"
        "jtmD;W!&;4)#usJ5vAwIL={W9s<iX17w#*)GF@WFfbpDe@UM#Ufk01Jb`4XUujT}#4O8yv$)-=W4)10*A1G}>_O4c-mi!3G9cly4"
        "Go*64ghQGh%u=;}g5rK{eOP{|@|dbb3MuZwaE<A!hlMZi_;v@|?*0b12X`W?&>?qpsRQ$GH6G|QImXp086;fL!G#@wU+@Db%rmbU"
        "Qqz-q=)E~UJACu;_-xiI%_Z7O7o+N2HAB<LFFa%lcrPztO0FBIXi!>0gsGNG`2e@KQ6Ask&gz-v4(huDeKZQH#5Q3h<dRq+151)}"
        ";}B$(6&YM#zIqQtE)I703Xm|e9cc0xF{G0nhcBpA<0pIIzP9G({bp`I%ViIK@B)q)Uk!Hn>o6BuNDBFyOZHl8rKgjtAx;hC)D94H"
        "`xFJPTwly5v%WndhiQkWaDa}mEqgew=L}w)f774Asf+a;TOq(#1r2^GW6w;7sJdbvBAmfC4%cNah=J}|u!mS(>$p!YHfC4&VGtjE"
        "^#RhyQG}h!@8Hk<<QWTslq_)3RL%L_^>y954AKU9UD5s6Vv>%WVRCQUtNo9@?6bz_x;_IpM*;3vp2wtlkGVsb9v_`NA%*;J){pk1"
        "pU;LSxLxc<#A-OK(I)$xS=R9-wGL(m@7JiY3p%b>+Z7Ws5AHT_kMRE;>5fJh?IU&`jo7vIo<z};gMB92i)}YYWx~^Y+G;u+YnnN;"
        "NAn}@xv>?_70_AG)9fhZ?*YmugL{7C$>YgltT7zk8^<u%nB4t6QI2%<7AzMcUGg6vk?dusrB=E%NLqLGpRewF8hP1^Whyl$^^O|x"
        "%yk=a$1}vi?JgJg`r%F{==bLOTW`<m@4AB>>%Gt$UR=6ec4y234NRHR0wigu)4oz7fh_cX^rx90DS}3Pn$lWx=&nk;1&+aQE2WV7"
        "o@umW89tuF@3i6+BFA1-Cebn`VDoIFfU!usotmfNbCC1`rm&u4sWc+akap+F&_H9Y<VCBF2905VXy3Y*MXK(ZxQwLDWc-@Be{?m7"
        "t&>SoX4+3g2r0jqoY%*(QuMZ07vjABH~U)o+gIwSA()7WHnKvm(00XVa^8KD>R{m5j07Hf@NfWdubqQeo^YUIxOrc(8sn^Wdm$-Q"
        "ZR$M>@-*-i*>EV=v-W%5PhR_KnKB*vKt3-*ShQZ>?C|@pbQj{yxG+5vVP8Y=FHlPZ1QY-O00;m803iUz0#RGu6aWB7RR91a0001U"
        "a$_%IX=ZO@VQgn*ZEayMa%FIDb7^#Ma(Q2KbaiBTE^v9(JZp2~Hj>{{mH)uwCo+-b$7GU;_q27J#IrYbmzk;YWb3LN7YiamiDQaX"
        "2-3DP<^O)&4S)nekdmG3ZJkmRizK?yX!ILR`M&=yP4leaUd4B%$m+7%d!lZVy|;rqtYURm7Bg?n3ogHVtVp~9zTR=~$M?T@A}*_}"
        "SbIPI_nU?9`(J+X<rmp@S5~#hsx=@GyuGnuVv}d9_WF}3i}s=v?N_mf?`c(Ty|~PC9^<saQxAVEo1*5G8n9zEqSO!H!zEHc&sJGl"
        "#w=g+?UpU#vPiSF9(Vaa?=L^R`T6qg$LQ_fe!jfI1=bON*zqdc@}eHWP**IjTS`6vf3oWGfo*p=pW&rq;One^eO0iX*pyDJ39BV)"
        "{!nYgF)PX<gCUVnkP)(O6x`T4g(3Lz3lBb5?|%94<7M>W@*i)m-n@HzHIujhp4FP9D}dtaw%l+Yb?D$L>~5l-SDYo0D4QyVG3C8z"
        "Z@0s1CT=4n5}(84aFpb_yyZpqyObzJbY24fJJ^9V<4L4O;7;Rz2~WM28#Dz<Pk{S>LE9dB^S?THej$m(NxdaW+ydtj1i7FkA?pKP"
        "#AO0<y!4wooxk$KP_M3Ts=|R-u~;L&_K^?%%}iabnBY-dZns&DH|qvuB<PDs@YPo-{0Y6ZtUUbYWrc#qE8*WL8p4hR_f=MNd+Bn{"
        "P%V`fh8tPbi|uWaRe`(^%a2t9a>*YuQAg!1U4{mX#7*2FJ|)d|CjvwbK!s>PG?<99Z26ky0uW_I0%(`#q4%}t|62Gs%3`96@VrO4"
        "lrZsU65nl%?=L=;Sz#^*1=Fz_y$0F~7=v4(NgJ+fuq{d}l5EXIZ3?AY)_qoQB%S4s7lB``{Lo_pEMk%5{Gy{+iJP*N3zP$mRgRr`"
        "zSywym*4u~V#6QQGlP_h|JG!%xPsqDWmE4O7d=@jsUdcXAS9#CioMhd;ADlY4?~kOmI>~m++GsUWmT0`;J>3OJQxmk8Z8jZQK|MG"
        "+BBAd-DY0Z6o2vJO<4j0<^f@zBy3>{NXqMY(`h|yj^IF=*ztG<7BSv%Eizw$I^i(2bBadrJ@0n)9QJL?Ylc#L(jt?{J&?Wq4CV(L"
        "<88}?=iWWf)*GO{yf4ryNC$@4$ZTzz^%{0PWw5oIrgbu#6t5x;gbP5BEi1AVHluH5ItB(veGo;or%|Y++&l4nxzvu_NTXUHBG~8&"
        "Gu4VBQH{dOQx1q@vL@HazC{9pobc)fWIT{}2o2r=K`?K4960J--<TVZUNNqD0_)OrUAL1gh>ua0+!*Gtm4&6;I9y}J?tS<>1VrzZ"
        "1BV0u1T2UjMS4v=R7V>F&4z+)VT4o<<b+;DYg<vuw;=k)O7SB#87QaS>OaGewfVttK{GXX#qTl!?rqr((YLe!)BnT%B-W*JT#_s)"
        "2&;X~MPQACjnqa%+6OY+JsaWu+fp&)Vz@4;-v3~E!<A-#MB+S>O+xK$GqGi$M+@JYJLq=&gcctvb1W$(LOMln!;Gc~AVh3UKguDj"
        "v>%8CcBFQFs=Xqt7{M2o;Sk2+CT$7<ch5*t6>1!Zaj2;oned9T_TH8SmxEf+nHK@wBhgD>BAJ|2MW{XNLBlB=kL2ScA{J>|0S^Hy"
        "OIvYK=65`3qnd$}kbl&3fngz_up>oD&^oIn<*=SSFYf_OeP33$OM2S7rT3};++yZ!vIJrV)QwLZBk3jBnwk2ioH=j>BPn>KB7IG>"
        ";2@5m3A_#rbB96510ZDNvU$;LF;C%v9jr>q@mtTUfe;Y}@2`qqfGEl{h}t|ro`bKJq=)SQdT1%o3TC_QZq?P^8i0W@&^{I_t9>SF"
        "X)QzL#_@}s73+qrIcl=M+t(ZL4oB%S<P(t`PZQe;jNyRd`!!93-)`C^>t~T`?d}O%5P{hdNU4N%Oic4ZoKQ2ckYGM>8*!_TnkIp;"
        "8WJTiH3kO9B$B^Ml7#~`It8>`ut^T1j~~mr1}&d3a{Lb;{!e#|xIa{C4{=mKRCA#RB+}atA3GrQQcxd;Nl|~bV3YOgEKKO^9lS7+"
        "qtdsiXw4Q5il=u~b_bgP0HZmigR6*bHa@-MQD<_8&sKRE--6|(nU`oKq{W0(%zOe#l>?aN8SI-_Y7i1#braIJA;6YBL@H~Swq8i%"
        "r|*%pTX42#m$AumEZq<El?B%h!dM-^mA!`;tKD*pv;u{{6)X#lkx7p40nOo9{%5xk%N+Qk1t<M*fYVqrLpDH++hdS2g1hjpKl?}@"
        "d`lNZsOJ;^i<)k!?~gLSCcC01Aq=j4eVpDikoIf8(>TE5-@q1hoIgQBU~}FwN90fzYzLWOg88V!7%P1;Vh)F%s!=Y2Ty#2OS@Oo3"
        "5{1r|HjX2bB(UwYHO}QfzFvA~W*!mqUF`*({hoR6uGE*05bc2=(-mdL9KTtMR#Af{_z7YO0SUQzkP8_wj0`lr$OSs$1%#tP@Ak~N"
        "4X(yzP2`-M;ogH}*Hu8P39m0+o}N0T&M9<NnWIUyi+I;YuyAvIuJ+Zeue25+@N4jy1xaD<qhv{(AhGnom-(L0@4)ydxs+qWnUd2k"
        "*p|{_Y>9J9iA%|nPgV&6V2xHsfNdB`3H8$4IA5*|7&p0agNr^c3bzaXn73Js0>^PDjX=&Ev8&3ML#%HN0nNoLC2l<k=|e6+#p62A"
        "GT$2k%2s1!kHKp@QirnDwU&#Ug@Dingy;g8auz~bBeQo|c#;!aGk;Q*a9pY!VLtD99vCv|j{M&mR<WYa3a%21NL7_C>_BHE1yM<8"
        "FS6;83vsjxzgx`xDp?nhn{XgKSwdxD0PsQW>MTYt1bG~&eZG&dh~wJ}XWOhGpa>xvWRdjY#yShMgiW49YQn0hiyT>#0K-g-t>F}L"
        "xkJhFfBE?O6=hvCeI$=RAWh|QL$huf#yK*l0`}w)BuDWrC`H5y;1v)Dfso8T^-Pbp-Wob&9KA$RN2DW}DUH#E&nhc=wOaYqS>JRd"
        ";G1U43RKrLdl+QnBX#>yo)$$SZBHvWunz-#8Rtzx4-YE3iugzH8@iPC)Kb8|Bwu&r&_>4XOnlpX*C^L9RcjtiR}G81;;GEb0qPp9"
        "*LGL?Sk|NKRkL320ZC3WX*ztCg7FyxGr~^?sEot_)-Vfafh6Qmeq$sOljkR-bSpr%*|O^P#FWdq%5rClkeO#6qWGHX#SZL(Vo^9V"
        "$S}U5Bd;1VqZqcm&K|sa!#(00ie%1F(&VXBU3vn(i4>kd7L_)IghK%Vp^0hK!U*p0G!U@%^}$-EBSP`ZJc(4SiiS(Bz!4?*C~3w("
        "+Ev#$L`X3oT2PZaBRG#8E1DI$9SWjaK3W6q09wlQ(l<BNL~d-3Dw6a~g2g|QSgM^s@MVWm{wiCeKz0(eWAUxM6#skn&DnR~o~Pfv"
        "V6WoU`HR)+<*T!^)pswFmyE5>zWs)$ar)xBR|7!AXy2Slb(~quIO|Y;aB&sdJ#u-&75@C@<A5wOwtDZhVg2e?JMZ|~2wZ6lmfP*D"
        "P@aN~0)o#6ugL3W_axwJ0vxOcZL__Y5!u)^D{$A(sWecPp(`qwZeUn4fUx(d%-%_(@<NW}tsC>4q+hPxmF7)kfJdjKqjdnonL$i4"
        "LWH$_F7tL|O~9%Lj8*^~*}vrij3R)vrH?k5k8UCBXXbLn#^b4U+02FdjB_McR&I3&R?T!Wkw`f=(ZV$vRCW;7_I0UP&wS65G8V04"
        "wm=RhT}md5WafHc<y0U<C#6rx9tarXo4Sn49LZLJdXlB7iqRsLFrcsc3!X~X5Ln{Dw1CiTwFjNdb9_QOFE%{!-qRvqEWTfyy`WAf"
        "4ptg?y(iU<a|Q?!lLuO|#v=63Y8wU&I*Qqkt}s5y*t#eM=1F!CuYAG!_iU#_>ygO3L(;Z4Y5Z|j3W08wjt<1EXL)~P@~%dIR8ai?"
        "BD`-;4b*|~Qd+X#n0ewdY%A&uBXXXO-?zXtVMO))BzjN)?8b>E1u3BE7wg|G=StNZ3+GgufFYSAos&*gZ>_Z?*$-rJ`*{x2I(xu1"
        "7yAGdbBpOvw<bKc0M_2)IUprWdPv}x6^p?K&2VGX2|INVz*Xdj_QC$}@aThvnuCC40$Us^URN1vo|Z2<%!CIx^Oj+MR;MVvbMSFq"
        "TKZ$L6AVo~d06=;@cN0)spt5f=bXlz<3GncnWv}I?no`||I!_mpucfOoNE1&Y@o9SZc+APwExmYC14y`Tt_52)Oay)woj`0J=ryB"
        "W=*W2=a9%h9#Kd_xoKqfto*ZdZ`$!lJnFcb@*s%c8-T1p33TlZ%xRO?o!=?Tw&g)D!)|(LYcOt|BNS6&9k!g~Sk$Ga<S}HL5Iryq"
        "TeEJ0t9Gl_<B-Y0L8=VOqctS+vOCX)1q)>z3=K`(Rf9Ft9>rkD1jnXok5G5@*4CV52{mTW8IpPOabq?|8YnyRQED)a0t_Lw8&QOl"
        "h+!Y0N#swu6l)@Fsafr@DWG>mFr*BN|0+e5iSw+xA^WoQZAe<fk#+RyvPRe2=V?wzv3hKofD9m_LvP-qnPk9wV|!HFq}LR;MR{My"
        "qR5+{I)1b#F(ei8D}+iaKgY@lF4#9&etgU)RnK^!^-Ik>l+A-)_NNw^6u1mKLaE!m_1ZHj_t>O|1$DN{>2(B_RXrEmo)!(L40K<S"
        "sc}QXRWFoodUcXMXmEunp%M6kLtT0$1eD~=j4(iv)uYsKc*ew5_1<lSZ!f$%lCIkse4)xPp;z6CxRzUL|4{}|<^&y!uzgsE;xvG2"
        "&RxBk<f)Zfg9WOaK$}4vH^}xw+oT$1OLh(mFwD8M&~@3G)UGk%a8#!2!;&B_VzvV<)HdDd2sIf#-4BD99O5t~$DJSnO34cxrOR^N"
        "U`G#450)$eDW}BJGb2_FP3@Tbe0%5>M_Ov^IfkPqI&3=HUF|n`v~Sb_<0{+Leuyr`cNL5kv7J^g@hn_AGW7`dz;%MI&1P2|lX2J5"
        "5=#}0xfUtg5Gi<S&*{tt#r_l5YGq48Bp`h&5*28xn^PpKrGcrS?%$4Sdd<aMR+R<jyuQ8mVV>3*y^#ngP=CANM90zn#I#k$u}FKe"
        "&Adm=H3Xg;EFj-lNB33$sY+C{!3aJ1C*qh6H@ipIW<-87fq=@`7YDP4$3FGDrrMQ4)=AB`MM%R*&#I`4i|F831`+hkcN-AZO3^7$"
        "q~46{-WI_>hkenL{7@H5+kWUMALQ^qe1CyKVZ)^t;`^Dpbo(N{A9^F55U7_1kI4zLLiflUZyM_Il}1BWZwv$N5Jkw;+5)bk|1KPU"
        "s3?>pj8-Conpq=6+Lj4L1YC$(>bewaayQW;n)mVbpGa160)Jt@-Eq7Obb%tEud|fiM$|9d5Sn)T-8D8Ms=HQWB@;jxK&}dO9J+os"
        "WR<~8ql;d>U?RI7b|uJ^dY}qz#sjA^C}tbO%)q223k*N(OgD%W_`r@?PEsv(WY&T8?WcNX9Hslt%cJ_v+wjF%a+&T(KceBB8oTbN"
        "Qh-08bw+&|eS?`1;0N!`uu*tN;^qc&wV0y#3~$>C*=mH&!PMEQ)92G5RsV1Y1&(^}NOvz*u+0)%|K13YmiJ>@f;C)S)*uvGjZA2L"
        "?}&zCGkbn=i=@^0ILLU%+8m{0#40V;#IU$dZ$FKUrj5CfGBYv+B`GqRC(GR^vR2X+!QGAHn=JJQcPT3x<EWO)7rSLgLieJ#-H>pI"
        "Q?eCTmMRC8MLHI@ZaP|J2#0E4pJ^%AuNDV`9@I%)8RQ$L+Pd-zhoUOhSa%Pr2?@*6rQ()OE{{-DrB};IrGJi6K2ITgS^WV;bRCmh"
        "3P@XH_FKD-!h6c6K)QP_2${dOgJpHKKkjj`raw7mbyfLBju~7PokOPy)6v$%c>-|`c+&+Owz_IPF;bS6xIr%xi$Q2`q|ezJ1ZG`v"
        "p6nr;0FQ_+7P~}ItVfQIbr6lGPRNqGC02X8_v9-<Eq(e&HWq)Y)(ysV@99a9aG9QC3K~U88Ap+h44W06PSG&z><3n@<ds0EdP%m0"
        "71Zi3k8>0A4O`j9N*U>j;3aV2)7zGF)#ch7Ov9KWF!1N|R<q}d1zx@1@umJrfZ?PnfO5>Lp7{NNgBg_tp5bwS7IU$bzsO+sn1_?o"
        "53pi{cQo-sn%c6Mf#LwnI;&kg14&_^&~z2}QW3-nzNl%`-VUIC?xqY7NkQn|$OTM{v}Ebi3uO^Frh^+sRhqeajr3U<sA+vJo7She"
        "B}a%8qmzt4mBlzPi{$n5nWa3@T%O@6G?YPUD!)1X{@Vjc#(`g)9};19KLLJrdU|>Q0VI=iWZeXoR|<tuH6K;vc{j)(fE`xrKzC_i"
        "ADBLQr^d|@)t*3poOVwjc(O)K;5t^$CQv*{)Y>MbiLL52_1bXHWal{ilesyL>R4uuV|WrTW#`zCkFuU<4jO8gE~y@*brfEQC=M?("
        "0z-xBxomeHwVPs(r6gV24<|87V~(mgZ%gizHr<Uz7hLC9aD4{6s^05GrFqFrQ+YZGe5}mqAknek`jKx9{R6u`ilVFX8P;+5yJt_r"
        "sdAsWQG)ve1Qt`s4neEp-7}$|Y)VEkJ;@FnWbA|sJ^TLr5Ee*WAz-?fs}oK%FV7BQnOMmo&|_5uv$ZY<p+AU+ilujIO#eqTbPB3c"
        "6Y7WT<r{Xb)BVdY!;IJ}ko!N#`sm-WK03cV{{jCEj_LFojwmI{ANnb=6i&x}vPk=%ucMFe{_FCsJUn%^gN6T;^?dUK{`(iNhv%-{"
        ")Onn>kuY?dsUs{|a<P7=W#;;gl9{6wF7*E0J3H!EMcx}{sH-MfW4gO-MD4A(M=3&f#|J-xH27JgBNyBcqK2&m6R&C>JA)BXkKlLp"
        "kRQS>gBSbfv-&-NCXPX$#PE1fKA-rZ0OwBwC^NzzvN|}Qd8yx3P}NVFrYy^Oa^Za@(8pifAHz^Tw|6SbEwIXnDrZqdD~uv6zC@8;"
        "q?DAay{P&25@(~@3`Rcx3s6e~1QY-O00;m803iUi`j2nk1polq5C8xi0001Ua$_%IX=ZO@VQgn*ZEayMb7Ns<W^Zg{E^v93S8Z?G"
        "HW2<EApe2Tz9fKOx^8QS76FztZjm-gv)SD+41+<FCz}f;Y9wXHE%M)YM@p9B*oo!>jYv^^?)AC5BuTD$29gPrJ8w(jmC4ym8*%Fx"
        "&n1JqLMx#>Q@Mw^<?+2KbIGj&H%OA?@Zj(uwI*YddoDD04jfZiVXSYq=+NQ8FPrrde(@FZlZh?i`0ya?v9nTXc}C!G4i6aqlzhWb"
        "thjKV`_i>HOlbwQHj2^~#??FY@9fLP<@xmN^7Fe-V|K!h;<*N=+Mr+Z?#1^>k0mI%h73?$NlP?Ls^)`YO^dO*>t{xSX08CCQYE56"
        "$Wmef;<hkoCstUqgq-IBl6e1d?<wH(eZbyw;a!FO8=!8=8@ukIS2!K+u^IPb5#S38QbqS|zTA(Fm@ju=l6!{2Xw4C)!o9KJ%`N2W"
        "D~5>&j(XaDYOGNCJmP~(o>TCLePGqq!HhMa%5#L8aR<YT39G?v3`j)Ixc3O&(Y8O*orUHC7Dh|38)z<h;qlrEc4E28Nhf;gz_a>&"
        "fJKK|%?rj$seDJ|pORi{{Xm4)ew4-eAVzC{VgdT-a1?>1fT3?Ot+-ZmGLARQST$EU*U>=Cq$vvR>MNd;?bSYqxD~ie`9)=wOCu0s"
        "{Ku#tc3D9V?@MLjVXQq7YF95_(@i=0cnMSVT(~%-ksHR{3hX*8m_>t|tiJ*^UwD>*oukGr{&^<at|*;aYYQn@pk!?CR`~JuDMl2U"
        "p^m}Uv5z64ok@Mh?|SURucsj#xLao#c&h{@`#;ebPuL!X+LdpW5RlD4N{qca7RQ_0SH@y)bBiQRf#4QMyBPYrAwrh5)ZGAV_Uc$i"
        "Cs4#_l>;Nf&SrWY6q|8zyW&<pYvWyMRiR-Td4-YsAgj-+bgsS1i-M93t!vl^$TH3}Ut_j)%B?HPt$py}Plfuv)VjJJ)yEgM@;H+a"
        "mJi~p8}zIF(1O@;!<$A{_0)k*hr${xai4w<X0{tLXW^Sp(372Y!dJXCG_w$gOQ@shCgiHg(Y3BDz%G@eN+0CF&#l2@-Ed#-P<ewJ"
        "9-t`~CRG}7Zbt7X%oJW_MBYK``V*#&DLO*`@Dy|r!0@R8KNY@2y2>CU%Mv73GIabJx41(FC9`!4u}O?@Qota~PWOo*mGg=UW)te4"
        "t8W6ht!pMKO`TCE@QY3qrt)fcI0tk9-^E`Oj?wP>DHanp$N%J-Maqo{fEv6-0jXI-c|$0&0&?2#T1j3%l&Kw@+f}ZK`fV}C*xM18"
        "hSJ}UjU972FWka-YC^G@jD`5$63Gy$I)|s|_WZ@q``a5_-uapdTlUd#r>NRhC)i419a=rt+>9thpY;_ixh{i;s**P?)_(EWUl==m"
        "mDfJH31^rS#)5r|tbx+a@a{#%?Jb*QG%~zb$=HSdFUL)`+LL{b+XQ;!ys1kcbt_k)5tMyl7S#8*_-%|dpmL#08Jep|yOEWs#iJ6`"
        "v1>Wq%8K2&**d^6FZ__m=22Q0qp7Sqk}1?Z8fijVSb{c4mi_Q?WU5ZmjUqemBaBu6U=2r|o4y{MdXASNMWETZvaC7Vi7iIN&mCOZ"
        "5}}zE+@n;)UN6L0d!+a{{(OCjb<8$({i*aDM_X)}pqY?bjI9-SeaBO14^WOT0sY2f8-nv;oZzNse@-vXagO^ICe#6qhvA8)g6avA"
        "r#<xD$2l<0m+xa6eQi7%U9Lyg8heDRi^gDK@3<O2akxFrLx+$@r}mgMqKqf&DHzAG965@!l+P>;g!sypLZFA!QZrlT^^i<gN=Vss"
        "N*&U4dPIFeFUmcwMqNYz&y@}f)HF$IVdxiCJYj8f{8fMdcXN&|9@P7IzJA;~Q7aSa<AkLN?zmyJ--j<3=c7R;zx!5yJ<w(a_UQON"
        "Y28DO#nJI#qf_&WerdA#`u<JrwrWhjXUYZB)h!sv0s9u!zh}wzMb|{h&O_nwqQ~O5g~EXO4^T@31QY-O00;m803iU|PzfTA0002J"
        "0000b0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FJE72ZfSI1UoLQYZOt(b!Y~Yl;k{DtV0ot(K-~}*fDNI@b(3O9;>t;pxIF=h"
        "jqMGe{)dPMG0$~(I-Wst^%jVI*+bZ3glZ&N@&e{lTLlPa<t_OJ=PzH3s%VBODnAB9?7O}TrU@nq$`lu$5)Hr5pwqp*$<r5*w5nBD"
        "sq)RSX(_ICP1o0_;T*;29E@<mKLoFR_Xbc)0|XQR000O8001EXPWSxqmIeR-KNJ7}C;$Keb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5"
        "XD?%Qa$$63WM6Y;bS`jtjTqZ*+cxw)K>mSKA1Vi~y9^r!xbxGt0)5y8*svGZ5NL^tiAtnEQcm0;|Gsm0kw{6-CZIKWZah49Cd;zl"
        "s3}jZB6Z(1?>Z?PV&oSl_*ZJU6oiV3l(Ov_X4pH{$?`=0`uI1grGkIGA|2&Qzc-JP9I0jv7ff%mEW5qAz2R*ql_4!Pr`wxa$rgys"
        "so_WBF@IW&#~pbsOWHiJwxyej8oH5njq(azo50aXS`}rls9{xs%C3nv^w_YX=Hh{=PH|zb3QR5*OFT2hYYrkk1yf5hjG|n?F-wZo"
        "+!PV`RrXFcyu1X?Qi_^C^okV@UbM1eaIcEhv~Nri*osccnm0_pDqDezG&fRRV(r4UiU_S(DOH8$zrBI8NC!9yS_JC`k-6?zxh81r"
        "qT;}X-0MN4DoP6Rv)bO?5ct6LKTDzemJ!-C1T_811o`xfCR`9WkEsttNTMl36bN4v#olRH8bA^WeyAJyNGc_}4zFV)3FRvJsGSmI"
        "H;;lm)o$f@8s2iFfgZ8@4DYT?cGzcDca1AYk{ujF1ttLv(e`E0Qo(DcO*n5JBi1sTfeE8IpPI$1!h|P+MJzdYXshjOhYb9$9fBXP"
        "Nz5=JSfd%qf0u%-3I6YS{EzHS*IQdXzc>gcxOL2$aKl>bKXk&z?(*R8h<R_x4<KWI7?CIk_>kRlMtt}|^c(hsrU0VN12cK%LD>pJ"
        "B2US!#O%$RH{_GhM)f78@@Mf3%=@=|1id}Hd<HlYeuSLa5=Vjc$D-W!$Ur*=fGj;_m2QK%E70<KCM{_Ce0K-utjTbtZMg5tdQNTg"
        "Yz_{tpJPJgUF9ikc<)@0?=uh04%cN(IIy@E9+e&DTvwxQPz0kcNDB>Ol(>G??rB_YXxA}O4Z&ZfQNPFN+F%G!QL=(pHu4Z?D^DWo"
        "K)xA^0{RZfu^c$;BtoGK!X5shP0xN+N~%2j=!#k?Pgq-d#f0JJLdp|`7F3a{hbHupbkR-bm5oyWc|0ZqRiR5lcx0ifVK=s=4a<0O"
        "zRx04WfmSaJYaOUiD9vEGxpR$WoenEwJsC1Wif@Xgj?^e7!{1FhjU`9#gt=VqFOU6?s*8iZBqOoD?%7-lY09yyQweS+esZ?I=Gsc"
        "Wj-8C()4*L4<QP+WIxB?mp{cpwkA&@_|Y1;4VeV}wB}C%YN&$ZhP&0WT#VMIp-Rqa7P4=tMEgJwRXSyQ8MiBMy_YmDCm{r3x?sUT"
        "?4>JadNrn4W(Nmw&j_j58TWy2oH=l63K0)+?tSA1Ah)){)}cM0*KxaAr@H&?o&UA7s@~oE?GeY^LPLxUAX=7v+c!2tci#`vo(gSP"
        "r+2njdRb@M6bzd4W%t9k-+G{?XEp;WhO<wB6Iv_)X151GB|<f0)$aaB*hRzBAA%9pi}!vA#2Lz6gd8tnB>VuShLvAB3C<wr?}+X@"
        "%m_^&%VC0qS;}cKPESxDy_BEGJ+{@+8UYd;Bfx|-&H-Tq9*C;ryfS_gU?@I&FdX{|f`2H2T|<QckM|u+<qU@ouiTw|D{M-}o4jEn"
        "w`Z@&dy+@`-X*Gr*~cA$v))8FTzzmU;o8nVt}PAV!A4#P-2Q*vQg#MvB~&}&4s)e*xModT-dxflFVwt3islkc9jGOGJQS0l>2!X}"
        "^rdZ?QM}AX9$U{1hhX%TC>-(#MOr(~jE<b?=;vP}yXy~*e6@FX@il1q#ghAwb+BLqe5trc2)z3?pa8{}Sa_E}@@Q}(mIG^r@LEyj"
        "DZnt4z|%+GV2@4PCIvbubS2auhRss|;h7~bgQfPFG{eY*)0Sva1)@?qBjKr0ntYXZAW+739F^PkrYQXNiXx>{RA}t@fCUCml&A27"
        "<4DUdM~Q<IYK_#kv!=y;2(1&ZU?j20GU)Ab8|S)BM6$jj<qVI@<EuhOim8kchfA7x+=3>lkfn^k5|n1ekQq^3#V;8mT=0ShZV{s}"
        "7$+mcl~(v4^Vi5VLmnp1_sM0`#11vMNZlQZ9iYceOcIwsAd0s!8MYXVN%%?oJvN92%jM+n7E!5+yD)xnC&q8=#`xKHscXw|=qA}4"
        "!N)G^YHoz`ecSy+@Th?=+9c`bR}3<-{%&UbsL_wh2p{$QaT(#`_8+YyieVB6i&#2N1d+3>nK42`bnPay&MNjMgr4tx_VHj17}}s6"
        "<tD8$<qfk#F}TL^Oq!&h^&Hdf&A(7f0|XQR000O8001EXg{`{f9svLV<pBTyDF6Tfb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD?)F"
        "b98BJY+-b1Z*DGdd2NxgP6IIvhWC^19jr{P1Xl*8Lt=uC4A32=Ip-Qk;=^$y`u6yG1r@28Y{&8c`HwMXcX!_dO%f>E2+;~M;P-?a"
        "5@vaTOadWI#DW751LIc>tnvN`g9c9EZM*A?F>TW}OD71iJIuLfO0W-}oQ)%<WV_<5_Y{ZoP}riC%J@R^igq|YGZW7*+h#zo3PnOx"
        "D#d$`RLZJI9TZHHF-Ykbscwh45Z{2HbqD0Dw`;h4{8#AP2J|TUw$2Rwym$;Q2-q0LJR7h{wtSY2U{Xs?q+s*#s{@YGR7GpfQOy$a"
        "xHvJ-Wg7}!-EXxtoW#|Ca|7lHeaJcoKB{i)gBpCf`^A^Q9z7|B`WJC7sOTS0$Q~K9!|n{RZkr!aO9KQH000080000X0Bmf7a~KW)"
        "0JARu04V?f0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rTWo>VAd0%jHZ(?OGaCx;FU2ohr@;yNQ1FJrq<g69j&7}voTfk|X!vRej"
        "G`Ze`VF+A_yXILdDI#TiTjamr%#akRk6qinI{}GT;>>WqXE;(6MIW}jELq)Xxs|_jE$iy#Z@*r=V@+GH#DZ0|X8Ky-ZPDep%s#yR"
        "$aKqPCEE1)v**v=)rC;!tgXAM$N)=LtZrF-E83Db*{0>qwMyAXAX0LzMax>D0b4QLcnip-`U~Sg`<AwU{)`zMwi4^Q6>P=xn>*eX"
        "3%2SsD{7&LtGwiL%ksL)yS5dTX5atwhYNNi+Dep6+>5+JYQSL>MaZVyHgyZa$gK!(>o&h0-lbK=IP6hP9;fTBA|3D&A>JaL=g)u<"
        "rPzDa*H87Y7e9ewcpC#WD5fh#at;!~RfsjqGFeHTWw8?F8sy<hoWn+`!CA1DS2yQOf-p;V`r2XQrMgD6agz2F62B(bgvh>KvXcM>"
        "FEEo5>^EL^;>WhF+c-iN4fwWG8g$H>T1im8NTYxzGMFNNzzP1DSxH;*sgtcxXyYP_-||w4F$vmfNgIw<RL)CTZD8>%Jz?xiaEIoa"
        "<1IUygD>lrXBV&z`g+e>z7^n0@laf2D6wtyE=CeY#i^{awcxq~UKe18TbYX`E!+vYxz?$91WS6?h@~O2E^Ds8It%0(i$)B+c??6p"
        "i?&uV?O1m@;KlIxzu*XXK*va{x`lu?4rW!hU`iLulk@~py)Qy+VN_XHyhRsLp~pd7YBa~&v@{;w|1hwIpEy~B<{8|*3HXDU@7l^c"
        "k0p|@7iObW*N!g9oPZOAZel_W`G$yJ%MIu+qZE}<Dgua@QptiHEx@BoneWa)?iK`ykeO%_T<rKeR>)jiks*i-8YLIVVShNhXx)`%"
        "^f^$?G^##YCL9eHTSx-4A(H6hix?e8k40QA?S2-P7rfD;0G2PW#*8kVYyoRvRaSv3E^rsfJ}5~kl{VT*ajD46l8TdLlrDg_RFIlA"
        "uW}I^u!T=h3FB2k<kMC)rPOhhf`U$y%j454(%Fz4UePDC)4!v|%+k&NrQ7{W_3YB;^C@V<u`pytA5eQ*2^9o0+KdQOYz1WsaIEGw"
        "LT9R58J)xd<s=}zn3MshLv+4mW)o}QlsAp2ikJv`s4v-t+HE1Rv@(bMC^yv>E1iVuB<>*8Y9_VuwN`Aw15I7#E3mt+>(moMjR{k}"
        "muwRX2L3Io9DcQ#2Lg@!TvuXPV*=0ckfC>sEZ~90zW%Z89Bt1qczENoD7mq}1Hp_D`bXfq6bSBqxPy-rYG2$I2P63BI^f0c!|nvM"
        "Adrbb5-^hsOPkIDe~(O1M&~v<1NV&3_^?2=a3*{ibU=a>9wHgfXkD-ZzV($5sAoCzECZzKwz1bu)on!y3Y#dk$4=&Hu6CL>)}m5%"
        "Yi{ES(r2HWhk$gWDA7XJwc1xgpbtWDxfVv(8u}^dUCb?VX;6tpL*zV+iTB<*5QeO)<fl$R2MI-1QL`#xjYHA=hSe){K_&|b8>x2?"
        "Ll85CD7r?uo)=UBkp(45r9*cwtOSXx2*~=#$MI3hiz4>WqaZiZy;xdFdSe{92jE&5)R_WS;-PyHuGenQuvKt>Ki08B53S+3qB1+I"
        "7+w|Fy!T{>KoI*{3;<Gkl$yF7cBBYSPf2ZifQBiB))%&6`e;;XU=y!_P19vtv8~%(Y&xWwDsCv4We1gFOToT?fXlmLP%o{jki@8U"
        "gVRZ!1LuWN;Bpk723d*r_7KFn6=MG?*?0ePBlu1B2sQ}rFfUe>$j+h<IHM`lND4~Lu)@h|T|1=>d}*YhF;_!UU?0!7Bew{ZKHznV"
        ">;PtP1C5e2ekIf<{SMY<RT=Kf@3TiRKBwt<Dtjadd>P8)T9&?lh00ZTjVa0s4=y^t#xueZ_V1!m%QNcf#4;Klt)W(zP<@F6WIeGw"
        "700g<9}k$k3{h-EOHtV^5{w7DwL2w@VjOm2-?Gz#wemyPH06$giy1aSFRj^w8bq#<BWOwxAe<}^`M!n%Z;0dj?O=j{ZTCpb!wonN"
        "A^NWS@F>FV-<Lw8d4(OwWn-%3ftF*`Y7o4U-QlRK@dmOW>QE#aQ-*0rM3^wqZ%_BbP6k{v9wQ{qs)OBn<6@m1t|SlYvBdTg&K$BT"
        "1a_s^a#mnVJXX&&Nm$IiT8+bFlP&J;u|r*+Tn!YN*!<i}$E#i3*oVlv{WYW8L-si=+C@KZ@{S@DwFcm+)Ho<0W%uso@#PRr7^WMH"
        "ds<B|3cmeUOa_5yY?p(8Tp-JW=hsU!Avx>-W`ir=3ytItL^KK@gTy|jEc;aPmNOME$X9At<=1UpVdpVO`Xe|@W;&zfXVV+jbT!1W"
        ")4JtMk+z+m=B+u(;tVc@oUzVwh*X@KSdbDrIH7{WuBv#Wu4_Gb*5$cS_=NQ7&JnxODHwANstvmCjfg3+jZY{CTA!KIq;Gq0o<q@("
        ";g#t+!{}HBry6&;6o+;grL(yWI^A?eR5Go}%!&5`TNRMg-{nlHP<T$%hLNi&eLGeK@@A(L)oqkO_oM;C!wj~KQo?})C^i^`YaUdV"
        "`dG#8+LLLZuP7+BDN|Co8FbzPa6yy@LrHh_Ug3_-^Lzdgq1Ev?1b?8QfCFv>+(wW`GduqBMfB)s>bRe$glE{9YGIEOICyufnNuOk"
        "jcS|+wmE-Q*JW&CQK~i0_ypNtGP{{-`&Wo9h_wEITN(8Idc}(t-N-#Y03>?t6G|a8q_{XqVTWxzH|juhBj~~^Y2lo02#SZ{8PWb!"
        "5bRNrb@wF;!)&h=!eDWZk5jfi#tFcmY@>Hl!HH<|vU;4Yf~}rtvJmrO%i)FAGg#9cY75^Wm7;CNwJiX7*ie{XCDdk0SWSwzqvpu8"
        "W3&)N(|qc}RZr2H5BsS(6K;|eml9j5ZCo4N!wK#obM|CrZR*-QNVCM&Cgm$=Yf>ENB<wqO^4~IqEs$^EygqP@f<<kXtMIrlnmPYM"
        "-{^^r1|;KDy*DZQ=>`GpNyy36`~Ghg{1MF*UXN|4Xbv)JBJmo{V;&G|3iyP%j6->FHe<zq)O-f7An2f3>`F|oqn80_Y2D#CF?xUT"
        ";lp@-A5ktg4I#h;=hQhsfWSZFCF=5^QKhM=D6N;NF<7rCWM<mO58L$#qux>NfQ?>Q(b%=d1~>i#GFz<Odn}q~_PSP|C-Q`j4-oy-"
        "^}x7qKM4$(A(`P!cFYb<%+gsmpX}n(`B>s<sF4nuPzO&q$Bo*q1B}cH**y8&INsoa5qR^YGyH_Q5_w$}M0I))V#n^FG)6NO9Akri"
        "jF-*%3zp4~@iJd)^7m;@ov`OpHrm(y9s>olfy}(KMsQ4^V@UsnMuu*cnJ+X{M&ECU0<b8~0<^&V{;Y?jCu3WaYwTGWELN7|Q{dx@"
        "!=E_XjgxJTXa3Q6+Z=32g3*RwvhRUe;Hti>l=D!Woo)FJq{8u;z=_0`k=@uYzr6bf$o2YwerL^B&_CgzG9A)BJT!2ZiGqXTx=AHe"
        "HDE={95M@NTxH;&8=Nb(VlD48NYxvC4W)se;*1GDX)v!8b$tVYvw#pO3Y>@F0Ln5GMbTRc{RsE;ZIm6GZ7e}1?s=}u9mp@>2o3#9"
        "SE1OTCXOHa@NH8<V@qH8SiJ}N2=mX7(Zn^2vI<&RjMxwgXP|>sA~Y0hFC~0-il5^Q;ql4oSEnampG}~>?uMK8kLeBj7l42AdV(A%"
        "*G`!a&cS2_wOkh2#H=eEoAP$&nm}<6d5}${cmc8<kF<461>3Xh$5DpJw}UT}raVn~>52_`WSuGeLJb@<vzEvA)BOmSBnKX2-8J-H"
        "-uoFlVsUJ?e_<!2!QwGHP1wtqlZ7O?JSU{9!TXQ+D6qvj7z6rKi?%?c1mOZ)TD3rAN%Kx$;gG(6v~XY79nr51N=$ET#;>!$seggf"
        "rX8)r$$%=xH&7w!hLy-r%dFL;Fm*>?bL|i)b#TZOvg}&nLj+%O+|eX1e1ZK&uIrqae;3;=KmPIT4gL$8pAo>TH~8-Wz}Z6puclj="
        "l~;uPkB{?9ufnCzg8@0I+6}~U^i2<vej09WUEFv8j7+PPM}<yc%S_xu0hjm+S2?%D=b&8r?j`_D2D}mwaDt~jzo&6W*%HfVZk>|P"
        "ZJ2MI46k1y@3;8hS7%?p{+ns<O;;kr4&A)`m#t|_Wbu?t5sK^GEqITEJo3wvD>$RB@DH+md1`J?%`NtoE8>DL;D?<Mm}tQNFHg^}"
        "9vI+<Y~TbxlU3egrR=}TsPEJ{sr?dz=qeZ*yYHfKW3u_EPr!cWe!*h<EzE*+C5p`Ez69|EzPrbXKCwMMcUwrzMbm0)$wDF;(uD_q"
        "Jv}3rz)zB>^HX<HU&p<_M~V49WQaKGkcYSZ5scDIY`%Z$C37N;a6ota$Eh<PcHtFy=Z8Ly_Spw$zcGStP(ymP`5llPTYK<^QhT)F"
        "^L^ORICQ+6Vn2Xp^oL*U(2sp!CIJT#0Yyq@_>J=xzth2H8y%Zmi>l`N%xnrT{D7Bsx^#zRs`ciaJdZ*yad|bv+W7D^In=v~6AsPh"
        "KAy8+F?t5JA5%*EnVzAyf>6B~)YP!}sJw1y4CFfj>ib;o>EDvu7f`5aGcouC7M6N9s1?XEnh*Yjk3rDPz4Eiz91NCWQJtP%=rw`+"
        "+LYQ3l{GlRhwQ`e<+j_}IR$Q$V15fq%n{uLSsyQYIbN2CaOs@vDp23jsgF_;K+*dvr0%QTe*jQR0|XQR000O8001EX8e_bhSOEY4"
        "lmY+%CjbBdb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@GJYGq?|X?A6EE^v8mQL%1=Fc965_=l?}y8?@(8&x_qN~VrcmIy^Im|)4A"
        "k!_^#_l20GL9I96d-v|%JwnJKTje0Mx}dC}6TWOxgL;7Nj>FbyExfVd8dZA-221%4+BgSo?>wOK0RnLm3OOMpj^aqQov|Lq#<E%d"
        "I3llmEWI))GzbU#sUeOkS%=~0(vI?HiQbxSpm7A_)8?YQ%V&*)L}zRQT6|0Wr#c!jR-P4B@|uG}4;x5VSr$hO(vJl>59+F(2XL%Z"
        "3ojdZgVA%4f7I4f&G(&_bY{Cobn=w13pU|16q;@joAsw>g>n*(LiciKt+ABsuAMCXc=MnJr_baoITj$W8so51RKp{dpn9VeoUq5*"
        ")BV4V(R7|W?eXlNae)J6i$$dGE#3{gzNHLaAq#{V-1GlJ_zi;B+6Ye>xAJC`rmF-NM@d%&y9e~dV#|0o&z~@4^!5W#O9KQH00008"
        "0000X04?Jm?XVC40GB@i04M+e0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rVW@c<@Ze?F{W^^uad9_=6Z``;M|35&!11q37$yv*f"
        "-dzhfTOdsww7|u8@a0+*+l9cDxVx(LsvfrYf_(SR3`vQUxa)Ot6>t}ekKu6mn<urN=l!+pibN#rhw}Cf`}zIHvv@1wyQ)l!mTg5="
        "iMnCyvS#g8u$;Hho3IaM##gK;Td^w3yC!`3;^m7s39njFpO!_oV*`zb@uul=LG!D$C}4V3i?-$|{@IjW9Say#3&!tx3Uo68^lnOa"
        "`@@XoWg;?Gb*n6md7GBSjK$Ez#G{Bid}qj+$g8sE8RKm$3TllxZ~$aH&qLl)9MHnlYS+JPd9%&ZRsa5<rY!oGniol#zkIQ-%bZob"
        "#gR<){RFS7r|Z&0H{;*m8jH1M9qVxoUkhNZDM6O18=IoC62){#NK=P__aa)SnLye!bd|2lm}eW2=R8b!%fq<LG7hu!oLf=xw0^5v"
        "UDHVZX|V~b5=6Vx6aAMj7(A;k&5}rui8giFRn1KHl~tSO>93-e%~8&abS;`zb$1nTBqB*R$^@^8L=BBdA;4X{e36JX!#zbwx`DYq"
        "ia5BE6l4>d>k|r_Ex-Kkdq0rf4{5ulGmw))l%em{D=z?~v8JsB&lM$n5W4ZUEACiYuoM95XFOje{7MZC;eUyLaenzNJ7M@wFk>sv"
        "3k(tiQ=#0QPXrY|faR_(^nk5+R9iB38dn_%QwaZ_XPwZb(=|(*v}jsh#Hwp1Q4@Y7AecO;RcKV0h!*he2O$vFVsTPS(zta{`E)0C"
        "R|8idl{_FpLHi7al%ufZTKLxZ!RG_3WNG7CW#L%BqP;vMhpx(mgDAE8Pf@myhMI=cAcbnyZ8R7u1$(kKW0<RVV6PSG@471hxVk4U"
        "b3kIcO0%@xskRf{*DnzPlVCpQo~eeBHlf7N{e_1bqBdQZB`N_ST`d>@Ser7t7y8g~Oqkz67brswwy&9gtWkX85zZtk@90$xUZ(}v"
        "$hjnZ#@61z-03IU-r}20R<3v!HLa+=SzeugpM3WEL!$X1Q3gPmN5j)bu=k+O*VK7k*JbUmy|;rSk+c2`C$jF~tT(K@7xhC8hjzt2"
        "5#i5X&<lDa1~z>!gsrtH&rUt|N?KV!KIoY_UI{@p!33UzH3e%sXP*W}3uwJ!)kuH9dV<r~bY<6Q0@`rkgc++4SA>la8<J?CZ%zOa"
        "Rd7K6?NT@;FpfNteM)tDJMItDXJ~|cx#KhDap^*v&~OG`&~Iv&l{~R`<n=K142A9+Lps(7&43>SGh<$F+i8S%W#L<nKuZ+W#RfX1"
        "RoaLa-{lnFB3JF&ZCR+%(s|-=$H=%#xJ4BkY|;_WGCx3h8Y9vs>f;0G6d6Env7A{VmcvodzLM6A^}>}AM=h5Kt5)L;PJ*Ko9SK&d"
        "D@t{&!sc!aebF*pSdB|-@1~_d(hkcpoDS|ZLcldh8oaYe4Cn;*)<IYkM%jGM4w;)v&ro75hJ`nB@Qt?w(d_g32G)o-HAl&IfVLhV"
        "VwmLxA?be9ipSQ+mN4n^s_``w;DjJ1E!z1dcw<pO$OllJdtJLe{ZoH#yuK5)OTo>+LNkCp4&_G8_T73nb3kS`*@$%r!**}4-@N<R"
        "6MH58kJ%}ltu%7I^$(@BXTMWs5_Ti4^S%)iYyA)OiPBJ;cDm&?sx1>u%qs}b!SO&4QHvN<YZqE38rDJy`n0g^aM7sgX6<!>GobMW"
        "j@N+oVV{e@X~2#&F!}%EhHX~HIHa>hqtp1la$YtLFw&GTm>hI3Kk^=A<i7%ygG~T=D4+!d1x+6S!;777D1@Oiig{8QoFxe?@O^1%"
        "jwUC`OhbY_AWI<$D)0m|*cPMs9Qq(h#7HeoD9>YSu=OGd2W^)d+#~0)d3jpYAe?>@K;G8|pLRUK%b~5=iGsONWrp*#p~U%$k#WQX"
        "SSDRMSYY>;7#dNUe19P`97>mKai2n1(r4PCH!L>Y$71AQmb4YK>DUtzkC67KIT(;L)T*<#yb}fL<XNwsK)6_^n|^blVXeN-6HQKL"
        ";*jW!p6iiD3Ejc{54)tvfYAi@PQLg|o!w;LcF?Ip&10iTNAPGU@x=E_t<<MJjuw_jTYEimV4j$5z~-;lAV6;td&&?CUwYkh*rS|R"
        "QC7-|&D?{y6xtp{y4ki(g!LgaE%|D^V7!;8Qw9zduSOi>Ef<~%1#fAAZ7+~a!iOPeL(2_7Lvx>m;?KD97=`*&7*%uPt|kXB;n<&)"
        "+LLYE@ySFXH>6|%QlLFyyP{?E*mDuYG<Vi73t0X&tt=7E_H3Mz9LPY-L;o@{3@DAu4oiz8m=Xd_2#m5Eb?&6iv=z`S7_P#0YKDMa"
        "zsr7aIo3lOCuNtSx8%h}_-61C1nATNvFx0^p6hj9OIgd=2Q~EX1^GckEkOF{Eo)_nB*oB<4=0(2Z6vI&V+{n3o?1HcPksH;$7oO6"
        "0(S;G1IWU`J(Ts`+#>^#fMbh=G!0a@Lx(MD+Y4auBHP0|u>&hfLdRCa6h6nSEAFt&HNlR?xnZn_Rdf|>v>0ZnRF6_zViwzgItrg+"
        "C;7gGKw7ab2Uj7rjfpwAw~8n<&4?z*$$U09AR`Ad(M3sKWWw`v_lQpwV&yzCw<*t7?lTw#?&KaKqx9f1$$^!!Ta|@DeftuG9<E9Y"
        "jV>+pLyZE&a!`xDyRSMdZHS>pItIzp8j24Rt`S+CiG&V9?(Hw{-YB$7qor`ySim!uY9^Hs7k{I43x*DvmcV)^PDWRml?|ktlwLtp"
        "M^*$3^2v;?5JNIoQvB(C^-<?sg)7>M8o64Ci}?K+wmieqH3W$`>pVQ4v2)c|n?dK1_=FAB<YJ_41!{Osn*G3J-UV0c1wQOs5I^WC"
        "EaU=LYN1yKy=4%#rLP1sC0%xH)wNnOKFxCs<4q%gupAxY8DP?b9^s}e#B@9$5j!Ek$$TlEw1ET~G}1R{1g^~v2)qS!LCmf&c!S|N"
        "Qou{R<P^v-q+!8MKN6U8h6%acQ<(9U$Qo3G(~I--AcWk+4;)kJ?Afcix!0k-iY5hu+Y`BhYc@gV)!ZCFm4nQLjR8YnqjOMll$<1V"
        "(5?MVi=n9?6~fCUn8_N*$Lf0+n<yH+fb?C7hg+g*CGPZ04TGF}m>S788l*Zk)S%76?q~D<tBuO_I)mUDRn<`np&je$hPZ$B?L*EV"
        "$>BG1I78JYsxsa}gL)m1-#76%?(!~^!TYzW(YnZ+Uy9T3RL29aa|ngyGP*I&(h5TMI-g&J{lZ`|B5^P3-TYFIqnh}*O^9TE@khN`"
        "%8DZldFQA*Gi{^F4RkeT8-`$gNN#rYQJ{>Kbyt56r|xS(ckpN=4;xH=BeD-vj=|m#&U-DLfavcUl(hv}MU8-*I8hdGQh0Lo0(gxm"
        "JqL96CpJ?DWgBYqXl8~)XNDvL_L^M`RKc8DXNPg<`7s3>9ffM?Vc^xmzl|AplJ4OE9q|H8siLC+K+9o^i4)-P{NUO*fB_&^h`efd"
        "a)C3X4Bnrq*ccY5fSWT(chFkLt(nEG0VpCzS7`cDi&`*uqE@^urdy?_FS|xHpj)$S%6Nu>(+(>pE$jq8c4*TzkMTk*-j*fME5rl4"
        "{pBXQdGp~KrFDD#^DiH+qxZKzesI)$o^|bK)g>0PB4N}bo~S}BE#VD=4<BFYDlH7aape04)BTI#5^|)==vGmk4R(ffQHOBU623`z"
        "{!i=fEvx62YUmn8l!%z`<`l*Ne*%2j{5{VaEm6FCG8XJB56-V9bnrD|JUEtNokJhT8cEmm2=@I5J|`#puAK<FBR=Vts@j1|7Zv{D"
        "kM}X2KPlE-?Jt|C(uzUiv$2h5g5=Wc9VG_abI_^B)}MkrP|>_m3t(0%kN%o=83cCLaZCUB=%64V+IHZ(h05oR`ok4{bwL$;`*!dg"
        "!vZww>Tg+ZxYFeaU2l+`+Z3-$Q2dMKPzf}QmstQ)WsiebEc!9yiwOcp?Z$GWo<k5C<Tsu)xrI4&w>p?@%dwk*v>}7O0$~ky+`!o^"
        "8TWD4#Dmvz6CV4;(c{Rv_~E-hTYSe{b{Y}z#E6n%SB5&{jd}Qly`jWq*N1FC$t(5Ki$Uz9CSkj)1l40P{WR;YM$!soAZ@J#zl(B_"
        "18ymOZR|WMD>wB0!|-i~MmY&9Q*wRS;ut5xuZM`1Jhd7ml?uCM%rvSZ+0{Ju7g>k1i+<zGYD5_llnZ;nKv<6fx|8MgvjR(f=UhRp"
        "|C9yN&(sn@_G&XSn5NMdo$L2v>7zo~DXL?FUxH>D*JVUesxXm=55^kQKP2;J2<xb)ebH&{5izJ$8WT_U^u>6QtI4ufN#L;HUP%ts"
        "Vk0SA!U=g*i6ZeQb%_NX_5#hoqr73IKk17_Np=}n)gnkTBb~HkX8vU2PDEv01S3C_EBoT0Po^dPCre#7juaY~7=oGy*h2YW)g6XP"
        "A^fN6KxjbPGQyyLd{K}KUQi*+%$+p6<T=GvH`&;TcU=|gC+scs98u(XPJ!nuiaUo3JAG#6EbL5+I<)Cy3OgoW?1cG})R_qk2%}a!"
        "zJu&5@yD3&kL$^@kdv}a4Zr`EL{utl2HreLh4Oz@h_6F-W28Nl)ljc-M#~B82XZQx#p0{$)HXp#U#Ll~+Td#aTz7HUKEV%ZgWo`&"
        "71NXK=~%}}iL_1*iX!aVG;2c2v2g+=+L4vo^!fMxG~oa&apDAzhq0&`CtfY)>C126-d2ot4f!vA=P{?c2bPJL?{xaF*hK&cG3y7E"
        "s9FUDid`;D+i+B_NSu`Vmftj6)(xgZGpS?YWy*J36_%5Jcf-6<&S(ne&=>Q=#|$%PNaQrC*yVq@xkZiZaX7PW>L}@$oRt+P)#bx9"
        "{q&CeR+5J5Y{?FAoEdARs<gfFnT(<@+eVdIX>rCwts|fUuKRSy^R&=N@n9aQ!pP{kbKN7NZG6mYqFoVh>k>c8nL05L{KAtloGdyL"
        "NF7leJGDYRiy;Y5cebS#39GX5N8%a>tjLOuX<~-{a>eO3pnh)@IZ-U<BPOW#$UwDc2PY8|dvD6H*lw1_r<Y}OXXMb6KSJ;(>(fn*"
        "O3JV1F51rq5YeDC4ICJLpMxK7q$|ng04~h3R36ul(<lQ|e0g&6hx3Ug&=rfN4s&6#@zu0W0o9LoxaUZCoxkFd>(K9YD5Ex$X<P@q"
        "g!k`a^bw4{BpA`r%_9c)LvX0AyC7&K+eZV@$lkTQrcP%6FIC2mtE!`)ebpW7{Py%=qQ0l=Pzh+d%VR1&`WcC8ujEYaaO>FgY(}N3"
        "<EtgA^n0Ye8C3YUG&63)aIASb@R^WH4hGDh5Ai4AeH^rvoW?=WOyY6`!DG<27g<i;zZ_ut@_3-?;TGs;FOz^sT{b>|t?h05)joG^"
        "-%O4}5ZygR3Q(Lj?wdXpf&D|6>GaR$&sXq}I-KIY{qe(3KU{yje&>DpuV$W%0*zw%*cVx&M3zL5N|JCaaah068IWu3+1#vWluc>T"
        "+BaGPXJOU2k0uf$FTC2dqOpGyb;t`p?fLE0!@WF~ZR`6dSJ%USv&Gk4K>;)z_X_;~7Jpz6)m;G`P?qXGkLb9cWPQa2*0V0J4*u{-"
        "3CEFvPHRMC5kX$?KTt~p1QY-O00;m803iUhHXr5_2mk<P8~^|#0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FK};cX=8aVaCx;E"
        "O>g5i_8uVrfmJWk)T3!yw1@2YFpEur0n#>TXLb*UAutx5up?0_NhOoX`oHgcq)3spY^Ok1mssNa{^m=TWmlj3hHtngTiWvGiS!-3"
        "J`pNvQZvnD%LP}OZ%Ef`-tsR@k~`TAy&~IA647aP+jXDF?>~Jgvn)G5J3r%X-$_k2UH{}?TB`4ZGu_F}eJKQ?iU@%RTiLbbmT$Wa"
        "ZSGjx(o#zbti@f~vbK{?RWG|+Mn(cR_;TI-_U@VsM&$yyi7meyB(rdp!$#WYXEob`Hd{J0+MgicmN!i0#pU@KfoIdTZ70Z*Jll7C"
        "vIiy=??iS<GO9Hb8eXcVleDsysva(~MF20U_yoWgKZYM^)8AA0`u%s|*R5pi3#%SGSt~gFAO|*c+&66I5a=oVp8e5rp`8Rf`^Ra9"
        "s*In|S)+%!AbF;q+Lmd_H`xNjo8Fu)@EQFKL<DWwC3(&o2`s;3uYUlfM=IN)uasszO8>*o1_VG_4oqyinn|QzGyAjHO~m=u$+KKW"
        "5(ENFLae}q{7$rB|7?vC9T#OH;Y_iz%+|4Y@p<(MBekp<>+wf!qo7#CP?^OC>h?!^Y@(`~guFbUU5aPb191BzI8J=Ia#pN50dgdY"
        "oK|8|ASM)o>?`pTZB!I@UfyT#)sum-i=$IAJxJjdn=<qPIM-COAVp1z<@V2lw7{=f{?ZBN!qgdgEL-Ir)49{Mupm>QL9xQnd_EZb"
        "i^Mj&AQnL+#mK*DGq5WuJDF!6#V64{3gV=FJ(f<(tqk~w!nt}<kgt|UN(zBo3ejx7TO^zEM`imJ#3u%hcoqcfrK|4`7lbRKh91i)"
        "s}D&Vm8b$y(1=)!QUQZ%6~P@9^FKiM&CT_dw;>UX)rwG}SWhLy0;u7_qIUcOS@l!T@=y#@ehV)F8o&#>vZrhEmRuYa%NPn_NjoS_"
        "pzTl~p}6J`Sll$bV=|N?pNJWziVLNw*u<jQCQvL$4kQ+2+jLYH#e{0xscDrZS6i@$b_7-<@;U}j4$0dk`PKpyr}Z;PUO~&ww0)<k"
        "*N}bg*#_;uB>9x~yQG9inL`-V4al^aCa^2We?yBs2c<ow8RVv=sKAh;RFnDfV`%8HMlGQE6((c60!hrK_09kXqRmV45)0qhEVGil"
        "@f=9qIUjHC(?a3e=K>YdkPHf|mGkka)Vc&WwbRUkPXqIhExO<#hif&2x?C)hZ&7^?0tzgdL@yGKf^|Md8)*(2X#z0m-+Q*%X77bb"
        "X>$FW(b-HFyD|~QWVRJG&E`?pw||A#h%>_vc$vQ9m|%Okj6^vU<Oz32s2qbnB#2^1hd6!y0YQm%d7km%;QE&p;x!84>bYcp4qP%U"
        "owv{_8W@X5hM`>zJ=1<81tBb;OUT?HK&nnzYkDmT7%QP5pGwVH<+O2jF^wY0q9l{jL`susUgw{Uizr+hAmTQl0fSs(D#)h!?u$=g"
        ">g7Q%9P)Tjfv`BH_1)ty;38ALud&tYr_=iU?#Md1ZH-dIks!Wkq?vJ`KAx7&-jz<v_D&rVj*<o_2&kl%RwASu<bunX9cYZvU3l4V"
        "PK=igeWL{w47UuNc<8nlvsh+0R6!JDD__PlKu~i5XK2i&#m0txk4r3=J+P~VSfsbm)Up<=RB)xm(Pp{VuiqqtNi-iDT_>B~4!o%+"
        "BiVNFH<%QBCpimld8_s5iyV&4Q`a2ldc~Nq4al|VuML&D!cnfpWrBiRcs=040mOsx^PWnGr5krlx?7A_wE)JOW~k9>zz6D?D$^e6"
        "X`+zI46)W4Wy&)z5>H~5_~X)5PB1C06Awb1LE{`)90+ypu?xc;6hEC*cE+W*5Neack5qvNhL%Y);Oa-dVNt##?UL4Y9&t^&Hhe81"
        "G^JLLT;F?-1=S5`j<cPOw{-GZ&)vGoGIa9<0Nv#<Q^(1iGf~^qJS)uw=N*QQ{rS(VNE60$bs=|(=NONF3r`o@Ay1Vy<e3TMqwu$="
        "z{TEIm*c$gkEv++R>h@?tNWOYD<jmpq<s(C<arYD&i#Hrvs%a5@xr{l+>0W2MNWs#{Yl43idgCKd;)wFehi=H;VDb^=9JmoL-~%e"
        "p7GCB+e0^<JY#fnUWrp}oVu5L`deP<X3Y5~7U6l{7#X5xm$q`v92+a(u4H8=)C!!9ZiqBpAH=Ni+@K2^0t-^T4Bf14oZc&Zodv^*"
        "vU(wGvi+G5qJtDQ*w}^lI%p^`OFm=mD~xLW)TYA%Z`KxZB|pBq`88c;j~BB|_rP$fY35RSiqWw}TimHfC;hh4h@X#}y`)KxBv!#i"
        "cy2Z5HiIi0L#f>Y*NkB9!yaVfm=XM_SF%ZUbmh?IeTmJg$pdeUgvJ<N?=4PiyvZ4S+oYcr+%)>qd_GKOw@1fZEvl{TzA)UzyA+K!"
        "RC@~CTwfhT!By41m4HsApTLQ@DYXMi_5MQzbKiX@vuQuJa%;wQdQC8kip(vrvi+4VKB%{CJ$#BIJXlZR<p1Mhr@gABf^V79V?@0?"
        "lE(6LG?3y-(s0LS6`kz6_x~1vf6-V&Z${oR^UO5`GZ9^&a&O56K}ODUSYzNd*(4mR76#ny%tR*yyQ9&r%<xP8@8U)GV%Xv}4)FnN"
        "zr~p~PGAs8{e6~;{G$AJL2@J8H{_y#dxc+|Y#mJ#l>nGDF<Zzdj^2CZmFE(gM!V_6#ZiEZXi-QS@bk0(0Z>Z=1QY-O00;m803iTu"
        "fR+o10RRAr0ssIe0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FK}{iW@&6`Zf7oVd2NzGYurE#h3^Lc2jNo!8@te(4}~0(9!i(8"
        "O>+&(SmPZsGjcU@H``xd&8C}@5}$-^z4!D<UDw@x_&7o$EUVB2dz9&bLa+$?5UGbP!~O31D^SspPzgc-6|!mqEP(%dg~}SW@4BvS"
        "+QthfcnqE$#+g#W-f{L}hL9E(g`4MvBqYlEeIfVW|HYyA9H(=~t(lt9mUTxs9dATfs#Duc<j;X0=vDWknDF?Y(FQDr(z8A&KT$3t"
        "gsdA-l$qYVyM(JxFa@V$+kibb*6mvTM>tTJ_gV&C3`zJ+GN9A&3pH>qL!s!ezT6K^xD+d@W1I$~njjjs?2dg6T}5|vn(Rz{V^fd|"
        "#dIuE%>t3~tb5~}4R?%2M_FJYd@4|AK{3U6gi7y4lDwyy##U$_6DdjQ;qJij`P&b0xS$If$P{hMnbnq9)@18bhDU2+jx_bJGrtsq"
        "q0qYvxTpuzZJ^uY$u#jqvGSu?PC)9t2>o0`JrQr&jn#_r@4iBw`|h%B{s2%*0|XQR000O8001EX=*qR9zz6^UFC72?D*ylhb8=%Z"
        "Vrgb?V_|G(Wo>O?FLZKYX>Mt5XD@PPaBp*IbZ>HbUvp-3E^v9JS>12rHWGhdVE+eJ_ae1gUmUc1*9CkJcgO<w&|=XIiUK}EU}VZB"
        "qOT(<$8M1Sy)*n0DO=8NtA|7)Im4OZkG~nxNs|1+`c87yi4$SRz7vY5?vb_pcdB^T654J_LshwF+jo6e^YTPKf4nDECkU-;BKeV&"
        "U0d<pAhdv7BuVo2&D%GX=o->fwXgXD;f>b(2|j(_1FyQ0);rcTbYXP3ad*Fcy8Gq7Ki&O&Uwr;__n*%*5=zigNhfq~%w}8gh;ngv"
        "q)lJ5OQKt<=%TBtnzyWwm2w)=w{HkMy#NJ*6>Q5DZ+C@YWhb^WH@&j2xti-!^r2=&1wBmkg0~7fvRM<v7<y}gV02r^ZV)9aME4|T"
        "MWZ#c!gUA22PSyM*|xAFn7E_^BF9^>@=_Eka4%@SDsB;BB^?l0VcLtP+p;<*TUODbR)rHQdeK$9W)iXR_6>A9%z@WFCN{QqQ6L&Q"
        "#GaKoVQmR|1?sgg-}=U)<_%Y??{9v5@B3}rKbJq=!fJdl`gnyZ7dxiXM07PvGV;xeB-+Hpj?wc{E*bfi)&si}q7!Mt4)CD^?^eG+"
        "w6tB|Ag)o887&A@c2s1A6Rs2cnQTn2vzu=vj8FqWHnnXF^FOr`IiXts2ZVsr+AObJ-HU-`8G3^qRzeU~gVAZ`T@5<}=U^S09xY49"
        "Hhh2-)F4{Ywe^WmGPD{}WXr3HR6`AG2k6)lwP%F4Qi-9|I2@#Fb#_?O7yoOMc!uvxFal{85PYoR=*1;#|D1VX1egIcTceEPhDk|x"
        "41>RZHc0qDl0h<&<ispitDwnD{<85DQ33O<9K;s1@9R@h?kRwP0q;=7f|$BVa*y-XM+(lFCEc->8S*U5EP7QD(wa@wp0|(Sj}b9S"
        "ZGgbp`m2qA2n3!|DLK#<_!zmXS4U9+`=jWQ{*%qiCLfhe{1`hfJ2V}2(X~AC&4f7G7tT++FU*LG+d}Khf+$_ngZ&EuIN^Poy{?W|"
        "UgP=$saczjre+&tk@5Q~k=S82$`9SpZmBpq+Az&vv!84-75T971_Akm0fXc`eywk`QMcHN<E&!%GH>g5OFdwD^Zu&20W(M8v1Ff&"
        ";Xy+KUdzb8VtU58ag6gUBk#x(21R`RrM4IwI>>uf-90%ggkgxwH9_jRhqjPt`8L_)Ara?6^l}@eY|K@zJ_wDl#P2EysB?U|04d9^"
        "sLRMjNui#yt>5NxA_x+g={V0~P9IKcn{RJ!Zv6Pe*yyZaY`a?iBOmjhD_kJm*NO?;95Z!KKq1reBt=JI%b2EBf|5NgA1h=NL7%K0"
        "%Nu;6-EA=oC1fgW64tX7>eOQ{UIEJo@zoSm9F~bLJO$9Yb0E8lllwiFL<}wP5eWQ$-~#whOe<c90vglsBV>`9?I@HqR{g>O>Tl$u"
        "VC;7WNe9wf&37CK3{uG+3m;mFPD)=8>#|3Z2L{G6vJ>6VgJNAcVqh0e<G-n7SY}DGrw0a{3w0LyhlxwlhLHo8b{bPc^z|Se9xN`U"
        ")utf@i}?J4Ff}-+2(UX0Nde7F0KixnEv$fC`$C%xc-Mh|=xTMrU_ZtBTxJ8)&eWdc%(*W9)5#uioEmJ|*;#<coNEs@aF$M16MFeB"
        "-ZSw1!}8Xqdvlb@-AggTe+?vn(bFjdQPw>LI16H-#Ay;49A3gni9_9oHOJWjW->lxgV~CA`d|foxyMgE({EAWhq+2n*Ntbbrg;1r"
        "aQ>z0Y_zkP@iDVSUo#Rh<;9CRZ*q)^3#zlY$wC0jCf05-DFg%9v1Cwej${IE6NWF7wFFL-Q2Van@F!9|;PNyMX=@{Zn`Y9ZdC>Ic"
        "<R$~yE+eiUB?A7(Q_cZrzjQ2(sN3?rP6hZUj}l1mo(Cz!zeVir4_@$6G!*J@Ce;-d(8muibd3o7*$!Sp&4DLZC-L&HvCZUA6fC?x"
        "I%)XpO^`*Nmr$iCX?a#Jf~znzgN*E<*8`~_9hWI&UuH@X8HhPiOFUY`1Z?Nj6Y6qEzPzcvns791W=RL1pt!m8xW$+s;V|*4j;|n("
        "K<^xZoS&6n<qmwZS1IL-Sp%O}nFAjKxPUZF!z@?e<1y&~;Q(@8eEFf*>!>?8#GMRZ{2lp*RthT+6t)E<)-|qH3ij>us5Jp>1ApP6"
        "GeDp%Z3;ZRV}TchJ~5nTj~5^Yyx+CFgo8(*K$lm=zN@$0pdxWwQ5yRS6E{#}*)|;Mh}e{KXA%>hz9I<+1{l8<lhegn)fyh96LW%@"
        "PHJAFZ~$pThvx2}IocD@KetDcXdA%5zp^coky=eT0Q#PtDaHs--?^XKOzJycLWM41TGQk@%HGP6^>GozRY&!RQS~T>8gVn(yfF9S"
        "i$s@^Xk?o`^7I0ZnJub7h4D1c=UeyDT%$a0&bx5>N1YuapyY0<@eQDkr+}{2O)c%^zEg$V)7u~3W0Qt|G9iwa^YS#VX2<?>Bw{ea"
        "D{O?^*}e88WvtI#T&GdQb8=inN9?032#WBqeO%b<5PdD+SBWdu$eB3mzGx}qapAq9b@nCZsdW;%yU~n%=CB(aUoeEo%Of!5Uj2A;"
        "fLH6`HA>~;XfDdbvd*97?HiC=Ke!#}cUF9b(9mNc73(FQxZSlv-<5l4u%EOnhi0ht<=pD~-1#Dh2D}wrv$A!+O9vHgczucsFi#|S"
        "{~Fxu0)uEjJcYp78`Vu~t3;lfhAF|zWR%g|qc|%ZNZce(y7bkO2^}TlvdsPtv6}y>A<fptmyIjrF8x-K+FvY8o3Htr9s>5jAsd}0"
        "IgN^P4*)Cy4Ej2?g@Fe)S>$P6W5*s6W@Bzz1NU1v^M~V%!I-l2OT^_=(a9#AL36afeN<ddH81hDu+X0T4iw%b2WXulHbyNzx)L|="
        "4Nr(sRD<>r#QrV3I5u!L|H5nM5ZB?y)XdTQRpk409|Z5?z4RYYO9KQH000080000X0NVjf-y{M603QSZ03-ka0CRF<FJftCZ)0I>"
        "XJu_|VJ~!YVQFq@Zf7raa$#w1WpXZXd4*F=Yui8&y&L==2AvX6Wz(h?K}t(vmmFe3V(3wfcW3RT`=RbfNJ4*nXJy$g#i2S#tKGLV"
        "^WMCvs%ky|*n-r7M1rT~8f4JV6reXKXy=Wj(7+QBNH5}S{rK%}b+`K7SlLunHJwZ+DtHSL6AFVdQUq;#AJS=Sg!GtpMsE(s5Bz()"
        "zR}7HG+VM38{wVO+XHU#(o@iuTzcwBoWn<fN^zlw8%x5J1Fd{|B$Tg<5PT?^rjxJ7nl3oZ8@cOs$aFcK410zB&5*mDPJkan7@T(7"
        "E?}Y-l<}D0H>|uPxE?-xoYf-uZi7kex~Sh6s@+-OHlWnZq(dRH%?6Y9u8WEK`0gExFaJUG+&UdOQVTJKvj|d&gA^Lu*LP8gQ0ypl"
        "fqrGSgKS~ry<x}_O{Beok`!<p?Yg~Vx~S5{{(9v0wEwiL9$e5_bXrkN?a=ia_8o3dg4f`4>T@bl+tC`A@Zrl)-PjGb+cKBNw#?Wo"
        "ZKwbunPS|bCfx8|j{z+u3b7pDEXtc{iJ!e^{hC6eM7y1Tn%&N?0G)*U`PJ-x{uykJ2{r_w*oGI%2Lj(GZMmsoR;U1bK(ZwoSnqfw"
        "NMSZ<6q&0AM`d;lX2&@r;6c;|R@oTXy|OxgRX9owmM}6iIWj8=j->Ys<`?O#wMY2}G|zHXL-nJ?EaleV*kjL}z)LbUJS>*$`b249"
        "e-ekwiP=&!rsTQ!15ir?1QY-O00;m803iS%Es_VN4gdh#DgXcn0000?Nkc_WQ$^icTXWmS6@Jf5{=*Ju+L2};+HsS#NuFd&j@det"
        "hm`Bd;}TeqTLV~R7ZAn%`ku?)KvJ?heXGZlSP=H?*>m~Mcg{kHFZ{XgWw}%0y)ISf)VJR}^Xfm9b=q{|)%;~4{w}*w+JkuY^5yqW"
        "ui6bm|MQC%`~7|{X`{KZ+ZPRQ=3adJ4K2ODUH!He*UKC6=5BejxL@2Y*W&HnN_<@37Gib#VRd)&@eMvM=&hT@`hK-|{Slvd5#P<l"
        "jjDC02hh;X{Zsfe^Cz1Lx06jHS|vL%z=;QCTPI{!iPCfxuM@SgVsxqyR`u3YV~LN8APZlvv>UA6jQ9&7ov3hERf)|(tX0WUzJm>{"
        "8Miy}lQ1>>py54Jj;-p3Q`L+;(XBN7!Rqa95N6*gD?lq)s)vJ+<FGSU|C@FUg3c`&c5)D~@779K*=^}1Z;OczYAYLYOVUsD7&|;A"
        "9YG0MlANFj*aF@SatyrTmDI}d?qJu!nx+uas_;WYiWhhyd^&a&EGtdhnl6a)Z;HJhb}WduoQt={k|zCVdjs*4j2m4aeKqr9W^@`S"
        "F7zd@HG5?XaH9opp&Gj5e-~mfqLko$d@qRM-;oupXk{n23cZLsyRqDP)e5oSDY5{B0c|Wv_{2JUje!MXE;Q&uUUEC#qo`WfVA5XM"
        "5@lcf@bXWWw4VVJvGt>*ac~3J9z6lhvdRUKAakQSFhr@rp%d|$rn&2XH{(oPz-s)Lon7YMhrei;2VIRQ&5Aruf&}#l8f%BTfbOk!"
        "j>4Zpj#GlXb0+H6uwe-~3HjC@2VHMft*iwB^xK*Y{~0%FO{L*{lCms%T6bkLl0hIVMP~-l=vJcw;817kVULmKXh(2P1x^UbNRoma"
        "exnGGQ0whz=>>34qb8VgxA_-D#i{zTJMc5`%h*u5)Yi1{%W@|>&^+b~L`diGRT+Ydo;3c)T1de*B6-F1kU_}NSs{&jjR|H*J%83M"
        "1O=P}9#2j^0W~=81B1ImL7dF3s<aftp`V`pld+#qfMO31No#^9jJuQ)x(i213Bd;RXKQ5z=<*<SBR5UJgbc0<#955SQu?rx2`d8X"
        "fi2+e2zzWVcw3Wk<uG9NAu|PigNzF}gL;&04|~H(z<r2%UWu<>_q~F>A0fLMv%kzuaigq$0Bb%dfmU&|<5<9L&Kts8Ac$f^1m!o<"
        "p%*(!&I&idWPm_o2*bU|ZI~MSoi2AdLV(u>s1T5>Ry}C)DaJI|$LEbuU}s~)PawnlJI}5l4eOwC5H#cj3EP>5as<}tt?po(Q{SH="
        "b%d{a0ttmUHlep5Mz(h$$@dk-s#fZhu&S4qLI!O^Cux;c%>k04`%K2$K-gfwbh1^K!TB1{$=0%@0#f8fG+OpVEwqp_b?*H)i1VHa"
        "ob~*1Lc|=-w~S`=2^gwvR0X9?Tto3(dBze&8MY8DgBPAUK#^t0fF;4e-c3YwH*Nq<Je~x4LUBgQlY%}kXeWaB6Y>$gP4)2EQj!xI"
        "glpO#Lw}>dpfy<I8ItpLj6lrdsb)bKbAkxhuu3(Mj@AH37wC$OY$%}i78iCzQO3@@Kwv^lEU%KW&{l&>i9=3s#j|B9!CAfw{7qT`"
        "O6mqzH$Xr^TvpE_V{!*KsMbxO9g6a(u(*^|Lj5Zo1y~oE(?n91TfWF)WfBaT$<cPeiObOu51=hu!cfmSKM@9IRZBevLrjkw0uxT-"
        "dR>lYbdbuew4V_SZ6Z;Tn<=Na9n}PgN%Rz1_iRWadd-$#bSd(@ozKoC^bsSEM-S=vWoDF{5izqp+E?sApN)dR0(z+k-$Czuqbwtx"
        "`eTGZYH;JyfYrI2BR|e^F+kI+x%d?c81DQg8X<5nu^zef`mjCQ1?5SXgSCPhDl#hx#1zn*2p=&AAd+xWAcMUcz&tS$0S=pLuaS>*"
        "rn{mLaByt=@d~(+-D1yW4zd{zSG85}lLiX-U`oXI(`xTKNZc}XYOn+{uE&@@0ly69`*8z{!IluTeIp_2;}cMu>#(CIo_gdx-=sB&"
        "2u<|*WY06zp1>yf=`Wd|K1jsk{|_f!z&h0rm~hak41q|2ijF&}OVRUjGbaMU2WdO`pok4dm8cgtb&Z4&s)K3(mHEE_GR6+<pNI^e"
        "R(twL$P*8TW?)x^eP!P_*sPikoKL1kJoCzxjnrUtem6fKn3AOACXQ&@K{h)lt)^tHEdWwzgp`hTF-L+6cL{y8>6EXQfQdj_VjDo~"
        "ju*r;afj?{F`R(O;8Z=o#~%F=>fl@~Y7B$aLODP$jPK~0L2p?Jxs~{90zTjSU!+==n7B30U6IxBB&8XVW%17t08%t^??ye~X&bfW"
        ">Ht;?>Svtc2r8cgJt`Kgy7RqyCa6p<I)sx2ue8KzK@Rh^X-Gy|6<S%JHK8?0`R2=EU|w7`FkjJ!F=nI-w621uAqu03K^(T2Re*-?"
        "=VGPuLDQUeZ{;DO-7zo$HtN8CrV#ZE8zj#lzl7aJ06q#RWIgbIlPdPK@8%l+1iBTee<G`7LDnk8?ygM(ofH#*0B1j^I(#87`M4uQ"
        "-4=C56}d+N2WvPd0$I+JQ)7vLPEISSavV1wzi^q2+itRLc@C7QH}+pR1mGEw#c+a_Lw6W6+<CfO0|dSp;V2Z_|1H@>v7qD4%WQM2"
        "21vOg5bbOZKph1rb$r5n@3?0QjskNoRnNjldx3CSDP&khPB|!Ihm?T+fIP}LqtOZ0;-AFw7*>N2Qm-;ZfT0B8NztgBgLcTu8XxAg"
        "njf34x<oXKvEbWA^!jXhcfWXZJA;IL9LVmNiGCYo>G@W9zGc90CUs8iOK!_$h1OMqFJ(n7YKlO0*0_jO5+}*I9FKN{7km~vMe#M("
        "&E=do^jYgt*nyBnNr#Qt<VwJ=NeK)L3by?@C@O=N$>7OK)A(}FYVk`B^pkL#WcqZXC3Kz82W#p!Rmjr~j4jTXS_WR2&CPsAamG^h"
        "aoP~efcBAH1`=&`h37m(4|O>6gN_^7hmz33-tV}9Lr^<o>&)%MFt`_wlSAk{QtyjA=13u*X@p{j$sIr}DYB~Y-xeFCJT!ALL3?kd"
        "uStr6UEshQlVedUhEtNNQe8E+f#gmiCm;<skHM)YFh^Dnvk)-F*~FsR7c^VU73_Eth-{pv??}!XD)l17a_LflS;EmIE%yjY@rTLP"
        "#wnjhk~?9=XGniGE<AHh#d5OiDN~=(r^s@sre5>#L<f@({9;~_K-@c@5GEB)uE-{3I`VWJRTKBdfSOKtVuN$kJskPv2kMgiodfr}"
        "i8H#n_}D>}b>uts2z!+p8)%Z5UB{$J$>ErEX4A23EcWCmENc&N&*P~O6T?kD-Ttq-I?u>R(JXRU22<TC-hHxwr5SKlye^{3voYMy"
        "VH$6#HN^Tt%DEBBgj1E`t0&Bf+%tYFCLla|1yjaeY76}!CBeaG3pI1<QI$FL6COvSShbawud<F?mU!jxBOqj8DGt#x<KC4aKpmK-"
        "XLnF&et(h294FqG$QEaDk@w&rP1y&$Iq*M%T0Wln<q*I@=@>Gp&9&-}QTzKUEcgywgC(C_3(UZ@cu|h}fUSX<lyFGzkkqW;6!>KJ"
        "8*(eS&)11FwIP`6uXAyu9kpF}Z(NH{z*fOHhnNvj+s%PHVCvqnJ<EWB+(~UunyeMcaUKm^()I$?#nHn_kDI@agVBkXFL6=^#b9>5"
        "7K`;vyuMy9*3rbD7WeP&KHiH@*Q?d_@_uo<7I&-s8-csG;(GbJ_{U;-1Dr_n9rPm(UR`z=O;}T9BiNKrG=r4^yAIGMlewssvS*MN"
        "EPB7Ve}7wm@t0SN<=fR_`Rna(x6AuN{C2x~^A7a9e!Y0Vxc{AE>FwfvdAsJX8Loqn57#Sj+{gFVEAipu>cidoma8$}Q#N>m0Y~Uz"
        "TTQncbPL2y<x$*#1Fh+;Mm|r6t|89w9g4&R#MvyCr<l$Gal#Wv7}t&<-<eX!PMgu*U-QzaH^1jS?S?}X{%sCVf?;vldo4G*p-cP)"
        "mSzDYsT)WYmI99&nxlf6(5>fVn{amqku>C^hfZx94J@rxmqomEFD8@XG)MYk>|ZdKz?*|cZ;1JkdRv@pqzjU;>42X@IJ%`jpDT>1"
        "le&Tfp6Izo)7E~lKrWW8+)i%_aC!JTA$_OddWCoV`2{#+7qDQyvOt={6Jxvt^il(G;}{P#$AP%zn|rMCsg~ol^>L$2CLJT{jQm7*"
        "-lZAXPUa35&tA+2z3}9X;n3Y0Q|)y#nMi(ynr-@C;_MMw!-%@qQa7XJLM5Ae>=FT@0(tJ+6ufM}IL>XtTPp_vgaL_!{&=<*gv5cp"
        "tR6I7&({8X637N`i14X}mmjj>r@6Q;u~?uf0v_Y$*Qt<a3G!)&Tz{IqCzq7ZUhxIOTJDVD`6`W2r&s<o&jo^4D?%86KhjZlrQ(D1"
        "JYe;Rc%TSWZHJ$`WHUB4ZWGiOX4Ckw7IB*w2t-Je`CbQ(gt_C}G96(xHYM-O9{VTmLZi{hs+r_;MEcIAYx0YiNYVWb3{8mrV?@`4"
        "t)wxrp>$!B0eu=fW%1xA&e&?{no%3`pVOc0Qz!dWs)inimjI8h&J5?JZ3&tJ6_1w8vU{{P-O%~LG$1uJ_1Nh0e6~29N^cIHjbvv!"
        "pcRr~Bl+KFVVUvEs7;9F+vN>b-E&{;`8U#?)2o-SeuP}Gqxn3@jrg;8t!q=t=2z9WGS}4cN*@GO#*wFACl>i9h~ffYh@bae{z5VW"
        "`$V5IDEZ-P;A^_`%?AZ~oDZY|CQ`#^HZ(Pnli^JOMLDDn@k`%1z6YANT;E=S+J4d3kg=yKi~ppF<YeQO(+_8Gu$;f}B-i>MiB@1("
        "Y`wKo6D7ihF_kT7L*aDYb_9R~yoXfpjz1p@{c@sa?(}G|)qem`O928D02BZK00;m803iVBu1g=*0RRAq4FCWb00000000000001h"
        "0RR910Ap`%W@%@0FJ*XeWpZh4Wo~qHE^2dcZcs}B0u%rg000080000X06Nxj(_jPu0F4U(0384T00000000000Du7u0ssJGZ*FF3"
        "XLB!dWpHnEX>@OLd0%sMb!2%iYIARHP)h*<6aW+e000O8001EXT`4=JW&r>I?E(M*8UO$Q0000000000fB~ll003ieZf0p`b1!sq"
        "VQFqIWq4t2aBO8RYIARHP)h*<6aW+e000O8001EX=Zh95djS9di2?usDF6Tf0000000000fB{7a003opVQp}1WpgibWpHnEX>@OL"
        "d0%sMb!2%jWp-g~Uvh7Eb1rIgZ*FW*O928D02BZK00;m803iV7V;l{_0000R0RR9a00000000000001h0S5{I0A+Y#ZE$R5b1!mb"
        "aBp*IbZ>HbUvqSIWO*-AML|SOMJ{b*P)h*<6aW+e000O8001EXdo9;C-~a#siva)tX#fBK0000000000fB^~%003opVQp}1Wpgib"
        "WpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1yMsW@KSzHa0jkHaRjdV>M=GIAu07WivT4Gc`3ZVlg-{H8nIjIAt<8Wj10nWi~f9G+{Df"
        "G&DD3F*iA5H8d`2b8l`?O928D02BZK00;m803iUpmUo)o0001s0RRAL00000000000001h0dWif0A+Y#ZE$R5b1!mbaBp*IbZ>Hb"
        "UvqSIWO*-hZeeh9Xm50LFEM3eWHDqpV`MQhGh;C`Gc{#1VmUT7H8W*qG&L|WVl-qhGc;ymWnwsCI5{(AW-&HlFlISnW@R@uFgGz{"
        "H7;s%Z*EXa0Rj{N6aWAK2mk;8ApqyD`R3pN004^t003zK0000000000004ji)(rpvWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKr"
        "VQ_P3Z*+4nGh{R|VlrenF)(FiW;HWoV>n@9H8N#6WHV$mH8(V7HDWU`G-YHlVKgvhGdMFcH!)^qV`VvJF*r9kH8MFiE^2dcZcs}B"
        "0u%rg000080000X04^)lo#6lg0E+<t0BHaK00000000000Du8K4*&pVcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8>"
        "G+|_9GchtXWnwfjGB!A7F=I3`Vm2^2WHvA{WH4nkH92NEI59FdVm4%9Vlp@|G%z?gF=An2VKF&oGBh$SYIARHP)h*<6aW+e000O8"
        "001EX7KqqE-v9sriva)tX#fBK0000000000fB~ft003opVQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1yYGHZWsiVK6l@"
        "Ic7CuH8C_XGc{&rW;J9mWHK~nHDWkpF*h_bI5%N9IW{?DFlAw5G&yEsIb}0sIWc57G&wG6b8l`?O928D02BZK00;m803iU(2MA%{"
        "0001r0RRAL00000000000001h0SgiU0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFEuz~WMwpAVP<79GC43|GGt<8"
        "VlZQ5G&nIZHZo>8Gi5S3Vq-TpIWsk8V`eckGB##6G%+zZVlgu|V`XDFIWB5*Z*EXa0Rj{N6aWAK2mk;8ApkQw?GfMr004^t003zK"
        "0000000000004jiauWamWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3Z*+4nHa0OZHDx$uWiw+rGBP<fVly!^H(_NsFfukV"
        "GdE>1W;rusGB;%~FgRp0GGsM5I5=ftW@I;FWi(?jH!x&3E^2dcZcs}B0u%rg000080000X0CkQ;P~ZRn0E+<t0BHaK0000000000"
        "0DuA66aWBacwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO9aIW{z8VL35mV_`L8H8VIdIWc8nG&eUiWMX4sW;ir5Ibt_q"
        "F*Y<cF)(5`F*#*nV=!bfG&MG2F*jvoVPa-3YIARHP)h*<6aW+e000O8001EX4ayOb-~a#siva)tX#fBK0000000000fB`@j003op"
        "VQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1!9NVmUc6H)1t4VPs@uG%_$TVq`fnV>dE0I50UlHZ?eAWnwaBVK_B7Fkxk5"
        "V>L50W;bMFVKgx|VmCN7WMnRCb8l`?O928D02BZK00;m803iTY&)elg0ssIQ4*&ov00000000000001h0jC!L0A+Y#ZE$R5b1!mb"
        "aBp*IbZ>HbUvqSIWO*-ia$#w1Uvh7Eb1rIgZ*FW*O928D02BZK00;m803iSbyiLPq0001N0000S00000000000001h0Wul@0CRF<"
        "FJftCZ)0I>XJu_|VJ}}_X>MtBUtcb8c~DCM0u%rg000080000X0P`loJ(C3h0E`j<02}}S00000000000Du9~8UO%ua$_%IX=ZO@"
        "VQgn*ZEayMUteuuX>MO%E^v8JO928D02BZK00;m803iTbT`SE`000130000Y00000000000001h0iPfM0CRF<FJftCZ)0I>XJu_|"
        "VJ~54Wo~pYUtei%X>?y-E^v8JO928D02BZK00;m803iTk@JS?d0RRA*0ssIW00000000000001h0WKi`0CRF<FJftCZ)0I>XJu_|"
        "VJ~54Wo~pYY;SLHE^v8JO928D02BZK00;m803iScl}$BB0000=0000Z00000000000001h0o@`10CRF<FJftCZ)0I>XJu_|VJ~5G"
        "b7gdMFJE72ZfSI1UoLQYP)h*<6aW+e000O8001EXmZN1%;{pHxVF~~MCjbBd0000000000fB|MB0047xV=rQ9W^ZF*Y-eR{ZDB8A"
        "b8}^Mb1!FQZEayRUu804E^2dcZcs}B0u%rg000080000X0JyAjh~okP04ECo04D$d00000000000Du9GCIA3)a$_%IX=ZO@VQgn*"
        "ZEayMVRLh3baO9fWo>O?G+$*jVlHZPZ*EXa0Rj{N6aWAK2mk;8ApnEz8iXAK001uv001ul0000000000004jiwkZGrb8=%ZVrgb?"
        "V_|G(Wo>O?FJW_YWpr~dbZ={AZfSaDa$jv>ZfRy^b963hb8l`?O928D02BZK00;m803iU%U1_O~0RR950{{Ra00000000000001h"
        "0U#{^0CRF<FJftCZ)0I>XJu_|VJ~B7Wn*h_Z)t9HX>MmOaCuNm0Rj{N6aWAK2mk;8ApnQMrm1`d000dR000^Q0000000000004ji"
        "?JfWSb8=%ZVrgb?V_|G(Wo>O?FJo_RW@%?GaCuNm0Rj{N6aWAK2mk;8Apj!h{%0=(0046d0012T0000000000004jip)>#hb8=%Z"
        "Vrgb?V_|G(Wo>O?FJo_RbaG*1baO6nc~DCM0u%rg000080000X096G%QiA{h0I~o803iSX00000000000Du7tH~;{1a$_%IX=ZO@"
        "VQgn*ZEayMWMOn+FJE72ZfSI1UoLQYP)h*<6aW+e000O8001EXXTKfly8-|JsRaN4ApigX0000000000fC0@o0047xV=rQ9W^ZF*"
        "Y-eR{ZDB8DVRT_HVPs)&bY*gLE^v8JO928D02BZK00;m803iUT(M`e70RRAL0{{RV00000000000001h0mnN40CRF<FJftCZ)0I>"
        "XJu_|VJ~E1bYU-Hb963nc~DCM0u%rg000080000X0OQt%K@9@{0LKXc03iSX00000000000Du9}Jpcf6a$_%IX=ZO@VQgn*ZEayM"
        "WMOn+FJo_PY+-b7axQRrP)h*<6aW+e000O8001EXK1jz6$OZra5fT6ZAOHXW0000000000fB_vr0047xV=rQ9W^ZF*Y-eR{ZDB8D"
        "VRT_HV{dMBWq5QhaCuNm0Rj{N6aWAK2mk;8ApqVF%Q3VV001&m001EX0000000000004jiBS-)Mb8=%ZVrgb?V_|G(Wo>O?FJxhK"
        "VJ~BKa$$67Z*DGdc~DCM0u%rg000080000X0Dkycq&5fu0KXgn03-ka00000000000Du7&VE_Pga$_%IX=ZO@VQgn*ZEayMWMOn+"
        "FJx(9XKrtEbZKLAE^v8JO928D02BZK00;m803iT2LM)Rs6#xL!LjV9G00000000000001h0gh<^0CRF<FJftCZ)0I>XJu_|VJ~E1"
        "bYU-Mb!=>3aBpvHE^v8JO928D02BZK00;m803iUr*cnlO4FCWSE&u=|00000000000001h0Rn#j0CRF<FJftCZ)0I>XJu_|VJ~E1"
        "bYU-VW@c<@Ze?F{W^^uac~DCM0u%rg000080000X0KrM{K}!b!0QM6A03ZMW00000000000Du9*i~s<0a$_%IX=ZO@VQgn*ZEayM"
        "WMOn+FK}UWV`yb_E^v8JO928D02BZK00;m803iT=t?Q7%4FCXpEdT%`00000000000001h0Zfzt0CRF<FJftCZ)0I>XJu_|VJ~E1"
        "bYU-YWpHnEX>@OLc`k5yP)h*<6aW+e000O8001EX0cAm7jvD{~>0<x@DgXcg0000000000fB{aT0047xV=rQ9W^ZF*Y-eR{ZDB8D"
        "VRT_Ha%FIDb7^#Ma(Q24Z*p*Tb1rasP)h*<6aW+e000O8001EX{};W?_Xq$0rW^nOApigX0000000000fB_`F0047xV=rQ9W^ZF*"
        "Y-eR{ZDB8DVRT_Hb9ZHPZE199E^v8JO928D02BZK00;m803iU+WD>Kr2LJ%o6#xJw00000000000001h0b0iZ0CRF<FJftCZ)0I>"
        "XJu_|VJ~E1bYU-aZ);_4X?kIFX>V>WaCuNm0Rj{N6aWAK2mk;8ApoTYH=F4I007$o001Wd0000000000004jiQqKSYb8=%ZVrgb?"
        "V_|G(Wo>O?FJ*RNY;|FDX>V>XUtei%X>?y-E^v8JO928D02BZK00;m803iTe#NZua2LJ%{7XSb#00000000000001h0f5i|0CRF<"
        "FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFJ*RNY+qw<WMwXJc~DCM0u%rg000080000X05_!f&KeW|0M|zV03-ka00000000000Du7}"
        "*Z=@?a$_%IX=ZO@VQgn*ZEayMWp-g~bzyXAZ*DJVb8R?lE^v8JO928D02BZK00;m803iVU1ZD}Y0RRB{0ssIe00000000000001h"
        "0fX!S0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFKA(MZe??GE^v8JO928D02BZK00;m803iV8>g^7s3jhFcB>(^{0000000000"
        "0001h0d?*G0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFK%IUX?A5_aCvlSZ*DGdc~DCM0u%rg000080000X0JanEsMi4i02TxQ"
        "04e|g00000000000Du8w`TziPa$_%IX=ZO@VQgn*ZEayMWp-g~bzyXAZ*DJea%E&`V{~b6ZgVbhc~DCM0u%rg000080000X02JkH"
        "TtEN-07C!(044wc00000000000Du94`~U!Ra$_%IX=ZO@VQgn*ZEayMX>Mj^a%FB~WiMY}X>MtBUtcb8c~DCM0u%rg000080000X"
        "04kA3%HRh80OT7004M+e00000000000Du7o{Qv-Sa$_%IX=ZO@VQgn*ZEayMX>Mj^a%FB~WiMxCZe?;|bZKvHE^v8JO928D02BZK"
        "00;m803iT%;)EM10RRBx0ssIb00000000000001h0W1Xp0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJE72ZfSI1UoLQYP)h*<6aW+e"
        "000O8001EXIFgjua|ZwbJ{JH0ApigX0000000000fB}{U0RVGyV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!0YX=G<*E^v8JO928D02BZK"
        "00;m803iSgX`>3i0RR9)0{{RZ00000000000001h0Z0x30CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJob2Xk{*Nc~DCM0u%rg00008"
        "0000X0LVsj7F7fQ0Hq5604@Lk00000000000Du8S5CH&la$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MTXk}w-aBpdDbYFCGVQzC~WpXZX"
        "c~DCM0u%rg000080000X0Qo1z8*l>v0D}nt051Rl00000000000DuAJ6afHpa$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MTZ)9a(XL4b1"
        "XkT!0Z)#;@bZ>GlaCuNm0Rj{N6aWAK2mk;8Apq*<Bm2k#001Qi001Qb0000000000004jiqZk1Ib8=%ZVrgb?V_|G(Wo>O?FKusR"
        "Wo&aVWo=?*WMpY>XLBxac~DCM0u%rg000080000X0JH>t$`S(r00IaA04V?f00000000000Du9m8vy`wa$_%IX=ZO@VQgn*ZEayM"
        "ZEs{{Y;!MZZfSIBVQgu7VRUJ4ZZ2?nP)h*<6aW+e000O8001EX#-x%R#|8iZixvO?D*ylh0000000000fB^>|0RVGyV=rQ9W^ZF*"
        "Y-eR{ZDB8MZ)9a`b1!XWa%W{^Utx4~Wo~q7Z*DGdc~DCM0u%rg000080000X03MT~bg=>e0D}eq03ZMW00000000000Du7(CIJ9*"
        "a$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MdZ)9a`E^v8JO928D02BZK00;m803iTvMhxiy4*&q9GXMY~00000000000001h0R|}n0CRF<"
        "FJftCZ)0I>XJu_|VJ~fOWMyn~FK%IUX?A5UaCuNm0Rj{N6aWAK2mk;8Apomc;kJJV002!F001Qb0000000000004jiMK}Qfb8=%Z"
        "Vrgb?V_|G(Wo>O?FKusRWo&aVaB^jIa$#w1Wn?aJc~DCM0u%rg000080000X02N1gdYc6R03;9q03-ka00000000000Du7pKmh=A"
        "a$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MiWoBh^Wo~0-E^v8JO928D02BZK00;m803iT_7_N^!1^@uN5C8xo00000000000001h0pvvi"
        "0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FLG~nbZKs9E^v8JO928D02BZK00;m803iUz0#RGu6aWB7RR91a00000000000001h0bxu5"
        "0CRF<FJftCZ)0I>XJu_|VJ~uJaBp*IbZ>HbUvqSIWO*)dc~DCM0u%rg000080000X0JZv$Z{7s}0ND@#02}}S00000000000Du93"
        "VF3Vha$_%IX=ZO@VQgn*ZEayMb7Ns<W^Zg{E^v8JO928D02BZK00;m803iU|PzfTA0002J0000b00000000000001h0hwn30CRF<"
        "FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rFUukY>bYEXCaCuNm0Rj{N6aWAK2mk;8AplPI{P30r002J}001Ze0000000000004jiYG?rf"
        "b8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD?%Qa$$63WM6Y;bS`jtP)h*<6aW+e000O8001EXg{`{f9svLV<pBTyDF6Tf0000000000"
        "fB{Bt0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPWNCABX>4p^bZKvHE^v8JO928D02BZK00;m803iTuY=d(c4gdhNF8}~300000"
        "000000001h0jqEU0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rTWo>VAd0%jHZ(?OGaCuNm0Rj{N6aWAK2mk;8Apja<yqj17005K%"
        "001Wd0000000000004ji3V#6rb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@GJYGq?|X?A6EE^v8JO928D02BZK00;m803iS^;~(v?"
        "5C8y|KL7wI00000000000001h0jGcg0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rVW@c<@Ze?F{W^^uac~DCM0u%rg000080000X"
        "0JJtA<`W120A?Hj03!eZ00000000000Du9VkpTd6a$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FK};cX=8aVaCuNm0Rj{N6aWAK2mk;8"
        "ApmTEmJ5ji004;s001Tc0000000000004ji@tFYtb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@JaZ)Rz1X>MmOaCuNm0Rj{N6aWAK"
        "2mk;8Apq#gwV%KU001u?001ih0000000000004jizMBC6b8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@PPaBp*IbZ>HbUvp-3E^v8J"
        "O928D02BZK00;m803iU|0ZiW{0ssIX1ONae00000000000001h0mq~P0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7raa$#w1WpXZX"
        "c~DCM0u%rg000080000X03a=r2c-@G0NW}600#g700000000000Du84rvU&=Nkc_WQ$<iq1qJ{B002q=N&x>E008&30RR91"
    )
    bundle_sha = "680575b706548604abaefa78808ee73b549a9d34c383dc3c2688da8c046a2a4d"
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
