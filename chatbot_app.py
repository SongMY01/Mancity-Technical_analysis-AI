import streamlit as st
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma

# -------------------------------
# 1) Setup (embedding_model & ChromaDB)
# -------------------------------
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
db = Chroma(
    persist_directory="./chroma_db_10",
    embedding_function=embedding_model
)

# MMR Retriever
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 30, "fetch_k": 30, "lambda_mult": 0.8}
)


column_definitions = """
[범례, 용어 설명]

Date,경기 날짜
Round,"대회 단계 (예: 리그 경기, 컵 대회, 챔피언스리그 등)"
Day,경기 요일
Venue,"경기 장소 (Home: 홈경기, Away: 원정경기)"
Result,"경기 결과 (W: 승리, D: 무승부, L: 패배)"
GF,득점 수
GA,실점 수
Opponent,상대 팀
xG,예상 득점(Expected Goals)    
xGA,예상 실점(Expected Goals Against)
Poss,점유율(%)
Attendance,관중 수
Captain,경기 당시 주장
Formation,사용한 포메이션
Opp Formation,상대 팀의 포메이션
Referee,주심(경기 심판)
Match Report,경기 상세 기록(보고서) 링크
Notes,추가 메모
Date,날짜
Time,시간
Comp,대회
Round,라운드
Day,요일
Venue,경기장 (홈/원정)
Result,경기 결과
GF,득점 (골을 넣은 수)
GA,실점 (상대팀이 넣은 골 수)
Opponent,상대팀
Gls,골 수
Sh,슈팅 수
SoT,유효 슈팅 수
SoT%,유효 슈팅 비율
G/Sh,슈팅당 골 비율
G/SoT,유효 슈팅당 골 비율
Dist,평균 슈팅 거리 (m)
FK,프리킥 수
PK,페널티킥 성공 수
PKAtt,페널티킥 시도 수
xG,기대 득점 (Expected Goals)
npxG,비(非) 페널티 기대 득점
npxG/Sh,슈팅당 비(非) 페널티 기대 득점
G-xG,실제 득점 - 기대 득점 차이
np:G-xG,실제 비(非) 페널티 득점 - 비(非) 페널티 기대 득점 차이
Date,경기 날짜
Time,경기 시간
Comp,대회 명칭
Round,대회 라운드
Day,경기 요일
Venue,경기 장소 (홈/원정)
Result,경기 결과
GF,득점 (팀이 넣은 골)
GA,실점 (팀이 허용한 골)
Opponent,상대 팀
SoTA,상대의 유효 슈팅 개수
GA,실점 (골키퍼가 허용한 골)
Saves,선방 개수
Save%,선방률 (%)
PSxG,예상 실점 대비 실제 실점 (Post-Shot Expected Goals)
PSxG+/-,예상 실점 대비 차이 (골키퍼의 퍼포먼스 평가)
PKAtt,상대 팀의 페널티킥 시도 횟수
PKA,상대 팀의 페널티킥 허용 횟수
PKsv,페널티킥 선방 횟수
PKm,상대의 페널티킥 실축 횟수
Launched Cmp,롱패스 성공 개수
Launched Att,롱패스 시도 개수
Launched Cmp%,롱패스 성공률 (%)
Passes Att (GK),골키퍼의 패스 시도 개수
Passes Thr,스루 패스 개수
Passes Launch%,롱패스 비율 (%)
Passes AvgLen,패스 평균 거리 (m)
Goal Kicks Att,골킥 시도 개수
Goal Kicks Launch%,골킥 롱패스 비율 (%)
Goal Kicks AvgLen,골킥 평균 거리 (m)
Crosses Opp,상대 팀 크로스 개수
Crosses Stp,크로스 방어 성공 개수
Crosses Stp%,크로스 방어 성공률 (%)
Sweeper #OPA,스위퍼 수비 개입 횟수
Sweeper AvgDist,골문에서 평균 수비 개입 거리 (m)
Date,경기 날짜
Time,경기 시간
Comp,대회 명칭
Round,대회 라운드
Day,경기 요일
Venue,경기 장소 (홈/원정)
Result,경기 결과
GF,득점 (팀이 넣은 골)
GA,실점 (팀이 허용한 골)
Opponent,상대 팀
Cmp,성공한 패스 개수
Att,패스 시도 개수
Cmp%,패스 성공률 (%)
TotDist,패스의 총 이동 거리 (야드)
PrgDist,전진 패스 거리 (야드)
Short Cmp,짧은 패스 성공 개수
Short Att,짧은 패스 시도 개수
Short Cmp%,짧은 패스 성공률 (%)
Medium Cmp,중거리 패스 성공 개수
Medium Att,중거리 패스 시도 개수
Medium Cmp%,중거리 패스 성공률 (%)
Long Cmp,긴 패스 성공 개수
Long Att,긴 패스 시도 개수
Long Cmp%,긴 패스 성공률 (%)
Ast,어시스트 개수
xAG,예상 어시스트 (Expected Assisted Goals)
xA,예상 도움 (Expected Assists)
KP,키패스 (슈팅으로 이어진 패스) 개수
1/3,상대 진영 최종 3분의 1 지역으로 보낸 패스 개수
PPA,페널티 지역으로 보낸 패스 개수
CrsPA,페널티 지역으로 보낸 크로스 개수
PrgP,전진 패스 개수
Date,경기 날짜
Time,경기 시간
Comp,대회 명칭
Round,대회 라운드
Day,경기 요일
Venue,경기 장소 (홈/원정)
Result,경기 결과
GF,득점 (팀이 넣은 골)
GA,실점 (팀이 허용한 골)
Opponent,상대 팀
Att,패스 시도 개수
Live,오픈 플레이에서 시도한 패스 개수
Dead,"정지 상태에서 시도한 패스 개수 (프리킥, 코너킥 등)"
FK,프리킥 패스 개수
TB,스루 패스 개수
Sw,스위치 패스 개수 (긴 거리 방향 전환 패스)
Crs,크로스 개수
TI,스로인 개수
CK,코너킥 개수
CK In,코너킥 인스윙 (안쪽으로 감기는 코너킥) 개수
CK Out,코너킥 아웃스윙 (바깥쪽으로 감기는 코너킥) 개수
CK Str,코너킥에서 직접 슈팅한 횟수
Cmp,성공한 패스 개수
Off,오프사이드로 무효 처리된 패스 개수
Blocks,상대 수비에 의해 차단된 패스 개수
Date,경기 날짜
Time,경기 시간
Comp,"대회 (예: 프리미어리그, 챔피언스리그, EFL컵 등)"
Round,라운드 (리그 경기 주차 또는 컵 대회 단계)
Day,"경기 요일 (Sat: 토요일, Sun: 일요일 등)"
Venue,"경기 장소 (Home: 홈경기, Away: 원정경기)"
Result,"경기 결과 (승: W, 무승부: D, 패: L)"
GF,"득점 (Goals For, 팀이 넣은 골 수)"
GA,"실점 (Goals Against, 상대 팀이 넣은 골 수)"
Opponent,상대 팀 명
Tkl,태클 성공 횟수 (Tackles)
TklW,태클 후 공을 탈취한 횟수 (Tackles Won)
Def 3rd,수비 진영에서의 태클 성공 횟수
Mid 3rd,미드필드 지역에서의 태클 성공 횟수
Att 3rd,공격 진영에서의 태클 성공 횟수
Tkl%,태클 성공률 (Tackles Success Rate)
Lost,태클 실패 횟수
Challenges,태클 및 1대1 경합 상황 (Challenges)
Tkl,1대1 경합 중 태클 성공 횟수
Att,1대1 경합 시도 횟수
Blocks,블록 성공 횟수 (상대 슈팅 및 패스 차단)
Sh,슈팅 블록 횟수
Pass,패스 블록 횟수
Int,인터셉트 횟수 (상대 패스 차단)
Tkl+Int,태클 및 인터셉트 합산 횟수
Clr,걷어내기 횟수 (클리어링)
Err,수비 실수 횟수 (상대 득점 기회를 제공한 실수)
Date,경기 날짜
Time,경기 시간
Comp,"대회 (예: 프리미어리그, 챔피언스리그, EFL컵 등)"
Round,라운드 (리그 경기 주차 또는 컵 대회 단계)
Day,"경기 요일 (Sat: 토요일, Sun: 일요일 등)"
Venue,"경기 장소 (Home: 홈경기, Away: 원정경기)"
Result,"경기 결과 (승: W, 무승부: D, 패: L)"
GF,"득점 (Goals For, 팀이 넣은 골 수)"
GA,"실점 (Goals Against, 상대 팀이 넣은 골 수)"
Opponent,상대 팀 명
Poss,점유율 (%)
Touches,공을 터치한 총 횟수
Def Pen,수비 페널티 박스에서의 터치 횟수
Def 3rd,수비 3분의 1 지역에서의 터치 횟수
Mid 3rd,미드필드 3분의 1 지역에서의 터치 횟수
Att 3rd,공격 3분의 1 지역에서의 터치 횟수
Att Pen,공격 페널티 박스에서의 터치 횟수
Live,살아있는 볼(Live Ball)에서의 터치 횟수
Take-Ons,1대1 돌파 시도 및 성공 관련 지표
Att,드리블 돌파 시도 횟수 (Take-ons Attempted)
Succ,드리블 돌파 성공 횟수 (Take-ons Successful)
Succ%,드리블 돌파 성공률 (%)
Tkld,태클로 인해 드리블이 차단된 횟수
Tkld%,태클당한 비율 (%)
Carries,공을 몰고 전진한 횟수
TotDist,공을 몰고 이동한 총 거리 (야드)
PrgDist,전진 드리블 거리 (야드)
PrgC,전진 드리블 횟수 (상대 진영으로 이동한 경우)
1/3,상대 공격 3분의 1 지역까지 드리블한 횟수
CPA,상대 페널티 박스까지 드리블한 횟수
Mis,볼 컨트롤 실수로 인한 볼 손실 횟수
Dis,상대 수비로 인해 공을 빼앗긴 횟수
Receiving,패스를 받는 능력 관련 지표
Rec,패스를 받은 횟수 (Successful Passes Received)
PrgR,전진 패스를 받은 횟수 (Progressive Passes Received)
Date,경기 날짜
Time,경기 시간
Comp,"대회 (예: 프리미어리그, 챔피언스리그, EFL컵 등)"
Round,라운드 (리그 경기 주차 또는 컵 대회 단계)
Day,"경기 요일 (Sat: 토요일, Sun: 일요일 등)"
Venue,"경기 장소 (Home: 홈경기, Away: 원정경기)"
Result,"경기 결과 (승: W, 무승부: D, 패: L)"
GF,"득점 (Goals For, 팀이 넣은 골 수)"
GA,"실점 (Goals Against, 상대 팀이 넣은 골 수)"
Opponent,상대 팀 명
CrdY,옐로카드 개수 (Yellow Cards)
CrdR,레드카드 개수 (Red Cards)
2CrdY,두 번째 옐로카드로 인한 퇴장 (Second Yellow Cards)
Fls,반칙 개수 (Fouls Committed)
Fld,상대 팀이 범한 반칙 개수 (Fouls Suffered)
Off,오프사이드 횟수 (Offsides)
Crs,크로스 개수 (Crosses)
Int,인터셉트 횟수 (Interceptions)
TklW,태클 성공 횟수 (Tackles Won)
PKwon,페널티킥을 얻어낸 횟수
PKcon,상대 팀에 페널티킥을 허용한 횟수
OG,자책골 개수 (Own Goals)
Recov,"볼 리커버리 횟수 (Recoveries, 수비 후 공을 되찾은 횟수)"
Aerial Duels,공중볼 경합 관련 지표
Won,공중볼 경합에서 승리한 횟수
Lost,공중볼 경합에서 패배한 횟수
Won%,공중볼 경합 승률 (%)
Player,선수 이름
Nation,국적 (나라 코드)
Pos,"포지션 (FW: 공격수, MF: 미드필더, DF: 수비수)"
Age,선수 나이 (연-일 기준)
MP,경기 출장 수 (Matches Played)
Starts,선발 출전 경기 수
Min,총 출장 시간 (분)
90s,90분 단위로 환산한 출장 시간
Gls,득점 수 (Goals)
Ast,도움 수 (Assists)
G+A,득점 + 도움 합산 수
G-PK,페널티킥 제외 득점 (Non-Penalty Goals)
PK,페널티킥 득점 수
PKatt,페널티킥 시도 횟수
CrdY,옐로카드 개수
CrdR,레드카드 개수
xG,"기대 득점(Expected Goals, 골 확률 기반 예상 득점)"
npxG,페널티킥 제외 기대 득점 (Non-Penalty Expected Goals)
xAG,"기대 어시스트(Expected Assists, 골로 이어질 확률이 높은 패스)"
npxG+xAG,페널티킥 제외 기대 득점 + 기대 어시스트
PrgC,전진 드리블 횟수 (Progressive Carries)
PrgP,전진 패스 횟수 (Progressive Passes)
PrgR,전진 패스를 받은 횟수 (Progressive Passes Received)
Gls (Per 90 Minutes),90분당 득점 수
Ast (Per 90 Minutes),90분당 어시스트 수
G+A (Per 90 Minutes),90분당 득점 + 도움 합산 수
G-PK (Per 90 Minutes),90분당 페널티킥 제외 득점
xG (Per 90 Minutes),90분당 기대 득점
xAG (Per 90 Minutes),90분당 기대 어시스트
xG+xAG (Per 90 Minutes),90분당 기대 득점 + 기대 어시스트
npxG (Per 90 Minutes),90분당 페널티킥 제외 기대 득점
npxG+xAG (Per 90 Minutes),90분당 페널티킥 제외 기대 득점 + 기대 어시스트
Matches,경기별 상세 기록 링크
Player,선수 이름
Nation,국적 (나라 코드)
Pos,포지션 (GK: 골키퍼)
Age,선수 나이 (연-일 기준)
MP,경기 출장 수 (Matches Played)
Starts,선발 출전 경기 수
Min,총 출장 시간 (분)
90s,90분 단위로 환산한 출장 시간
GA,실점 (Goals Against)
GA90,90분당 실점 수 (Goals Against per 90 Minutes)
SoTA,상대 팀의 유효 슈팅 수 (Shots on Target Against)
Saves,선방 횟수 (Saves Made)
Save%,선방률 (Save Percentage)
W,승리 경기 수 (Wins)
D,무승부 경기 수 (Draws)
L,패배 경기 수 (Losses)
CS,"클린 시트 (Clean Sheets, 무실점 경기 수)"
CS%,클린 시트 비율 (Clean Sheet Percentage)
PKatt,상대 팀의 페널티킥 시도 횟수 (Penalty Kicks Attempted)
PKA,상대
Player,선수 이름
Nation,국적 (나라 코드)
Pos,포지션 (GK: 골키퍼)
Age,선수 나이 (연-일 기준)
90s,90분 단위로 환산한 출장 시간
Goals (GA),총 실점 (Goals Against)
PKA,페널티킥 허용 횟수 (Penalty Kicks Allowed)
FK,프리킥으로 허용한 골 (Free Kick Goals Against)
OG,자책골 허용 횟수 (Own Goals Allowed)
Expected (PSxG),선방 예상 실점 (Post-Shot Expected Goals)
PSxG/SoT,유효 슈팅당 선방 예상 실점 (Post-Shot xG per Shot on Target)
PSxG+/-,선방 효과 지표 (Post-Shot xG +/-)
/90,90분당 PSxG+/- (Post-Shot xG per 90 Minutes)
Launched (Cmp),롱패스 성공 횟수 (Launched Passes Completed)
Launched (Att),롱패스
Player,선수 이름
Nation,국적 (나라 코드)
Pos,"포지션 (FW: 공격수, MF: 미드필더, DF: 수비수, GK: 골키퍼)"
Age,선수 나이 (연-일 기준)
90s,90분 단위로 환산한 출장 시간
Gls,득점 수 (Goals)
Sh,슈팅 시도 횟수 (Shots)
SoT,유효 슈팅 수 (Shots on Target)
SoT%,유효 슈팅 비율 (Shots on Target Percentage)
Sh/90,90분당 슈팅 횟수 (Shots per 90 Minutes)
SoT/90,90분당 유효 슈팅 횟수 (Shots on Target per 90 Minutes)
G/Sh,슈팅 대비 득점률 (Goals per Shot)
G/SoT,유효 슈팅 대비 득점률 (Goals per Shot on Target)
Dist,평균 슈팅 거리 (Shot Distance)
FK,프리킥 슈팅 횟수 (Free Kick Shots)
PK,페널티킥 득점 수 (Penalty Kick Goals)
Pkatt,페널티킥 시도 횟수 (Penalty Kick Attempts)
xG,"기대 득점(Expected Goals, 골 확률 기반 예상 득점)"
npxG,페널티킥 제외 기대 득점 (Non-Penalty Expected Goals)
npxG/Sh,슈팅당 기대 득점 (Non-Penalty Expected Goals per Shot)
G-xG,실제 득점과 기대 득점 차이 (Goals minus Expected Goals)
np:G-xG,페널티킥 제외 실제 득점과 기대 득점 차이 (Non-Penalty Goals minus Expected Goals)
90s   ,90분 기준 출전 횟수
Total Cmp   ,총 패스 성공 횟수
Total Att   ,총 패스 시도 횟수
Total Cmp%   ,총 패스 성공률 (%)
Total TotDist   ,총 패스로 이동한 거리 (야드)
Total PrgDist   ,총 전진 패스 거리 (야드)
Short Cmp   ,짧은 패스 성공 횟수
Short Att   ,짧은 패스 성공률 (%)
Short Cmp%   ,짧은 패스 성공률 (%)
Medium Cmp   ,중간 거리 패스 성공 횟수
Medium Att   ,중간 거리 패스 시도 횟수
Medium Cmp%   ,중간 거리 패스 성공률 (%)
Long Cmp   ,긴 패스 성공 횟수
Long Att   ,긴 패스 시도 횟수
Long Cmp%   ,긴 패스 성공률 (%)
Expected Ast   ,예상 어시스트 수
Expected xA   ,예상 어시스트 기대값 (xA)
Expected A-xAG   ,예상 어시스트 기대값 차이 (A - xAG)
Expected KP   ,키패스 (골 기회 창출 패스) 횟수
Expected 1/3   ,상대 진영 최종 1/3 지역으로 전달된 패스 횟수
Expected PPA   ,페널티 지역 내로 전달된 패스 횟수
Expected CrsPA   ,페널티 지역으로 크로스한 횟수
Expected PrgP   ,전진 패스 횟수
Matches   ,경기 기록 링크
Player,선수 이름
Nation,국적
Pos,포지션
Age,나이 (년-일)
90s,90분 기준 출전 횟수
Tkl,총 태클 성공 횟수
TklW,태클 성공 후 공을 소유한 횟수
Def 3rd,수비 지역(Defensive Third)에서 태클한 횟수
Mid 3rd,중원 지역(Middle Third)에서 태클한 횟수
Att 3rd,공격 지역(Attacking Third)에서 태클한 횟수
Tkl,태클 도전 횟수
Att,태클 시도 횟수
Tkl%,태클 성공률 (%)
Lost,태클 실패 횟수
Blocks ,차단 관련 데이터
Blocks,총 차단 횟수 (슛+패스 포함)
Sh,슛 차단 횟수
Pass,패스 차단 횟수
Int,인터셉트 횟수
Tkl+Int,태클 및 인터셉트 합산 횟수
Clr,걷어낸 횟수 (Clearances)
Err,실수로 인해 상대에게 기회를 제공한 횟수 (Error)
Matches,경기 기록 링크
Player,선수 이름
Nation,국적
Pos,포지션
Age,나이 (년-일)
90s,90분 기준 출전 횟수
Att,패스 시도 횟수
Pass Types, (패스 유형)
Live,오픈 플레이에서 이루어진 패스
Dead,정지 상황(세트피스 포함)에서 이루어진 패스
FK,프리킥 패스 횟수
TB,스루 패스 횟수
Sw,스위치 패스(긴 대각선 패스) 횟수
Crs,크로스 횟수
TI,스로인 횟수
CK,코너킥 총 횟수
Corner Kicks, 코너킥 유형
In,인스윙 코너킥 횟수 (안쪽으로 감아 차는 코너킥)
Out,아웃스윙 코너킥 횟수 (바깥쪽으로 감아 차는 코너킥)
Str,직접 슈팅으로 연결된 코너킥 횟수
Outcomes ,패스 결과
Cmp,성공한 패스 횟수
Off,오프사이드가 된 패스 횟수
Blocks,상대 선수에 의해 차단된 패스 횟수
Player,선수 이름
Nation,국적
Pos,포지션
Age,나이 (년-일)
90s,90분 기준 출전 횟수
SCA, (슛 기회 창출)
SCA,슛 기회를 창출한 횟수 (Shot-Creating Actions)
SCA90,90분당 슛 기회 창출 횟수
SCA Types, (슛 기회 창출 유형)
PassLive,오픈 플레이에서의 패스로 슛 기회를 만든 횟수
PassDead,"세트피스(프리킥, 코너킥 등)에서의 패스로 슛 기회를 만든 횟수"
TO,태클로 공을 탈취하여 슛 기회를 만든 횟수
Sh,슛으로 인해 또 다른 슛 기회를 만든 횟수
Fld,파울을 유도하여 슛 기회를 만든 횟수
Def,수비에 의해 차단된 패스로 슛 기회를 만든 횟수
GCA, (골 기회 창출)
GCA,골로 이어진 기회를 창출한 횟수 (Goal-Creating Actions)
GCA90,90분당 골 기회 창출 횟수
GCA Types ,(골 기회 창출 유형)
PassLive,오픈 플레이에서의 패스로 골 기회를 만든 횟수
PassDead,"세트피스(프리킥, 코너킥 등)에서의 패스로 골 기회를 만든 횟수"
TO,태클로 공을 탈취하여 골 기회를 만든 횟수
Sh,슛으로 인해 또 다른 골 기회를 만든 횟수
Fld,파울을 유도하여 골 기회를 만든 횟수
Def,수비에 의해 차단된 패스로 골 기회를 만든 횟수
Player,선수 이름
Nation,국적
Pos,포지션
Age,나이 (년-일)
90s,90분 기준 출전 횟수
Tackles ,(태클 관련 데이터)
Tkl,총 태클 성공 횟수
TklW,태클 성공 후 공을 소유한 횟수
Def 3rd,수비 지역(Defensive Third)에서 태클한 횟수
Mid 3rd,중원 지역(Middle Third)에서 태클한 횟수
Att 3rd,공격 지역(Attacking Third)에서 태클한 횟수
Challenges ,(태클 도전 및 성공률)
Tkl,태클 도전 횟수
Att,태클 시도 횟수
Tkl%,태클 성공률 (%)
Lost,태클 실패 횟수
Blocks ,(차단 관련 데이터)
Blocks,총 차단 횟수 (슛+패스 포함)
Sh,슛 차단 횟수
Pass,패스 차단 횟수
Int,인터셉트 횟수
Tkl+Int,태클 및 인터셉트 합산 횟수
Clr,걷어낸 횟수 (Clearances)
Err,실수로 인해 상대에게 기회를 제공한 횟수 (Error)
Matches,경기 기록 링크
Player,선수 이름
Nation,국적
Pos,포지션
Age,나이 (년-일)
90s,90분 기준 출전 횟수
Touches, (터치 관련 데이터)
Touches,공을 터치한 총 횟수
Def Pen,수비 페널티 지역(Defensive Penalty Area)에서의 터치 횟수
Def 3rd,수비 지역(Defensive Third)에서의 터치 횟수
Mid 3rd,중원 지역(Middle Third)에서의 터치 횟수
Att 3rd,공격 지역(Attacking Third)에서의 터치 횟수
Att Pen,상대 페널티 지역(Attacking Penalty Area)에서의 터치 횟수
Live,오픈 플레이에서의 터치 횟수
Take-Ons ,(드리블 돌파 관련 데이터)
Att,드리블 돌파 시도 횟수
Succ,드리블 돌파 성공 횟수
Succ%,드리블 돌파 성공률 (%)
Tkld,태클 당한 횟수
Tkld%,태클 당한 비율 (%)
Carries ,(볼 운반 관련 데이터)
Carries,공을 운반한 횟수
TotDist,공을 운반한 총 거리 (야드)
PrgDist,전진 운반 거리 (야드)
PrgC,전진 운반 횟수
1/3,상대 진영 최종 1/3 지역까지 운반한 횟수
CPA,페널티 지역 근처까지 공을 운반한 횟수
Mis,공을 운반하다가 실수한 횟수
Dis,상대 수비로 인해 공을 빼앗긴 횟수
Receiving, (패스 수신 관련 데이터)
Rec,패스를 받은 횟수
PrgR,전진 패스를 받은 횟수
Player,선수 이름
Nation,국적
Pos,포지션
Age,나이 (년-일)
Playing Time, (출전 시간 관련 데이터)
MP,경기 출전 횟수 (Matches Played)
Min,총 출전 시간 (분)
Mn/MP,경기당 평균 출전 시간 (분)
Min%,팀이 치른 경기 시간 대비 개인 출전 시간 비율 (%)
90s,90분 단위 출전 횟수
Starts, (선발 출전 관련 데이터)
Starts,선발 출전 횟수
Mn/Start,선발 출전 시 평균 출전 시간 (분)
Compl,풀타임(90분) 출전 경기 횟수
Subs ,(교체 출전 관련 데이터)
Subs,교체 출전 횟수
Mn/Sub,교체 출전 시 평균 출전 시간 (분)
unSub,교체 명단에 있었으나 출전하지 않은 경기 횟수
Team Success ,(팀 성과 관련 데이터)
PPM,경기당 평균 승점 (Points Per Match)
onG,선수가 뛸 때 팀이 기록한 득점 (Goals For)
onGA,선수가 뛸 때 팀이 허용한 실점 (Goals Against)
+/-,선수가 뛸 때 팀의 골 득실 차 (onG - onGA)
+/-90,90분당 골 득실 차
On-Off,선수가 뛸 때와 안 뛸 때의 골 득실 차 비교
Team Success , (xG 기반 데이터)
onxG,선수가 뛸 때 팀의 기대 득점 (Expected Goals For)
onxGA,선수가 뛸 때 팀의 기대 실점 (Expected Goals Against)
xG+/-,기대 득실 차 (onxG - onxGA)
xG+/-90,90분당 기대 득실 차
On-Off,선수가 뛸 때와 안 뛸 때의 기대 득실 차 비교
Player,선수 이름
Nation,국적
Pos,포지션
Age,나이 (년-일)
90s,90분 기준 출전 횟수
Performance ,(퍼포먼스 관련 데이터)
CrdY,옐로카드 개수
CrdR,레드카드 개수
2CrdY,두 번째 옐로카드로 퇴장당한 횟수
Fls,파울한 횟수
Fld,파울을 당한 횟수
Off,오프사이드 횟수
Crs,크로스 시도 횟수
Int,인터셉트 횟수
TklW,태클 성공 후 공을 소유한 횟수
PKwon,페널티킥을 얻어낸 횟수
PKcon,페널티킥을 허용한 횟수
OG,자책골 횟수 (Own Goal)
Recov,볼 회수 횟수 (Recoveries)
Aerial Duels, (공중볼 경합 데이터)
Won,공중볼 경합에서 승리한 횟수
Lost,공중볼 경합에서 패배한 횟수
Won%,공중볼 경합 승률 (%)
Player,선수 이름
Nation,국적
Pos,포지션
Age,나이 (년)
Premier League ,(프리미어리그 관련 데이터)
MP,경기 출전 횟수
Min,총 출전 시간 (분)
Gls,득점 수
Ast,어시스트 수
Champions League, (챔피언스리그 관련 데이터)
MP,경기 출전 횟수
Min,총 출전 시간 (분)
Gls,득점 수
Ast,어시스트 수
FA Cup ,(FA컵 관련 데이터)
MP,경기 출전 횟수
Min,총 출전 시간 (분)
Gls,득점 수
Ast,어시스트 수
EFL Cup ,(EFL컵 관련 데이터)
MP,경기 출전 횟수
Min,총 출전 시간 (분)
Gls,득점 수
Ast,어시스트 수
FA Community Shield, (FA 커뮤니티 실드 관련 데이터)
MP,경기 출전 횟수
Min,총 출전 시간 (분)
Gls,득점 수
Ast,어시스트 수
Combined ,(모든 대회 통합 기록)
MP,경기 출전 횟수
Min,총 출전 시간 (분)
Gls,득점 수
Ast,어시스트 수
Player,선수 이름
Nation,국적
Pos,포지션
Age,나이 (년)
Premier League, (프리미어리그 관련 데이터)
MP,경기 출전 횟수
Min,총 출전 시간 (분)
GA,실점 (Goals Against)
CS,클린시트 (Clean Sheets)
Champions League ,(챔피언스리그 관련 데이터)
MP,경기 출전 횟수
Min,총 출전 시간 (분)
GA,실점 (Goals Against)
CS,클린시트 (Clean Sheets)
FA Cup, (FA컵 관련 데이터)
MP,경기 출전 횟수
Min,총 출전 시간 (분)
GA,실점 (Goals Against)
CS,클린시트 (Clean Sheets)
EFL Cup ,(EFL컵 관련 데이터)
MP,경기 출전 횟수
Min,총 출전 시간 (분)
GA,실점 (Goals Against)
CS,클린시트 (Clean Sheets)
FA Community Shield, (FA 커뮤니티 실드 관련 데이터)
MP,경기 출전 횟수
Min,총 출전 시간 (분)
GA,실점 (Goals Against)
CS,클린시트 (Clean Sheets)
Combined, (모든 대회 통합 기록)
MP,경기 출전 횟수
Min,총 출전 시간 (분)
GA,실점 (Goals Against)
CS,클린시트 (Clean Sheets)

"""
# -------------------------------
# 2) 시스템 프롬프트
# -------------------------------
system_template = f"""\
당신은 맨체스터 시티의 수석 전술 분석관입니다.
펩 과르디올라 감독님을 위한 경기 데이터 및 전술 분석 보고서를 제공합니다.
2024-2025 시즌 맨체스터 시티 경기 데이터를 바탕으로, 상대팀의 전술 분석과 팀의 경기력을 개선할 수 있는 정보를 제공합니다.
지금 날짜는 2025년 2월 21일입니다.

펩 과르디올라 감독님의 전술 철학:
- **점유율 기반 축구 (Positional Play)**
- **숏패스와 빌드업을 활용한 압박 탈출**
- **풀백과 중앙 미드필더의 역할 변형 (Invert Fullbacks)**
- **전방 압박 (High Pressing) 및 카운터프레스**
- **라인 간 연결을 위한 포지셔닝 최적화**
- **상대 팀의 대형 변화와 전술적 대응**

당신의 목표:
1️⃣ **경기 데이터 분석을 통해 경기력 향상을 지원합니다.**
2️⃣ **상대팀의 전술적 패턴과 약점을 분석하여 제공합니다.**
3️⃣ **펩 과르디올라 감독님의 철학을 반영한 개선 방향을 제시합니다.**
4️⃣ **객관적 데이터와 전술적 통찰을 바탕으로 의사결정을 지원합니다.**

📌 **추가 지침**
- **데이터는 표 형식으로 정리하세요.**
- **단순 나열이 아닌 전술적 시사점을 포함하세요.**
- **질문이 모호하면, 추가적인 설명을 요청하세요.**
- **답변은 한국어로 하세요.

{column_definitions}

"""


# -------------------------------
# 3) Streamlit UI
# -------------------------------
st.title("Manchester City Technical AI")

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []
# 기존 채팅 메시지 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 질의 입력
user_question = st.chat_input("질문을 입력하세요...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # -------------------------------
    # 4) Retriever로 문맥 검색
    # -------------------------------
    docs = retriever.get_relevant_documents(query=user_question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # -------------------------------
    # 5) 프롬프트 생성
    # -------------------------------
    human_template = """\
{question}

아래의 문맥을 바탕으로 답변하세요:
{context}

📊 **출력 방식**
- 표 형식으로 데이터를 제공하세요.
- 전술적 의미를 분석하여 추가 설명을 포함하세요.
- 필요할 경우, 시각적인 비교(예: 상대 팀 vs 맨시티)를 포함하세요.
"""
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

    messages = chat_template.format_messages(
        question=user_question,
        context=context
    )

    # -------------------------------
    # 6) LLM 호출 (스트리밍 출력)
    # -------------------------------
    model = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        streaming=True
    )

    response_container = st.empty()  # 스트리밍 출력을 위한 공간

    full_answer = ""
    with st.chat_message("assistant"):
        response_placeholder = st.empty()  # AI 응답을 동적으로 업데이트할 공간
        for chunk in model.stream(messages):
            full_answer += chunk.content
            response_placeholder.markdown(full_answer)  # 실시간 업데이트

    # AI 응답을 대화 기록에 저장
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
