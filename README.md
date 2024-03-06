# RealEnglish

## 💡 Motivation

OPIc, TOEIC Speaking등 영어 회화의 중요성이 대두되는 흐름에 맞추어

영어 회화를 준비하는 사람들이 자신의 영어 발음이 자연스러운지, 어색하지는 않은지 체크할 수 있도록

원어민처럼 영어를 말하고, 사용자의 영어 회화에서 개선점을 찾아주는

'영어 TTS 시스템'을 만들자는 생각에서 출발하게 된 프로젝트입니다.

학습자가 읽을 문장 text와, 그것을 학습자의 스타일로 읽은 음성 파일을 받습니다.

저희가 준비한 '영어권 아나운서 데이터셋'으로 학습한 TTS 모델이 정확한 아나운서 발음으로 text를 읽습니다.

학습자의 음성 파일과 TTS 모델로 만들어낸 음성 파일을 음성 유사도 분석에 이용하여,

학습자의 영어 회화에서 개선이 필요한 부분을 제시해주는 시스템이 바로 Real English 입니다.

## 🛠️ How?

원어민 아나운서의 영어 음성 데이터 👉🏻 [[Voice of America]](https://www.voanews.com/)에서 제공하는 영상 사용

영어 음성 데이터에 대한 대본 생성 👉🏻 구글 확장 프로그램 [[Transkriptor]](https://transkriptor.com/ko/) 사용

대본 수정 및 음성 데이터 변환 👉🏻 Python

음성 유사도 비교 및 분석 👉🏻 Python

TTS 모델 👉🏻 [[VITS]](https://github.com/jaywalnut310/vits) 모델 사용

TTS 모델 재학습 및 음성 생성 👉🏻 [[Coqui TTS v0.11.0]](https://github.com/coqui-ai/TTS) 라이브러리 사용

## 📚 참고 문헌
