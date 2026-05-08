# Cammand — 제스처 스마트홈 제어 시스템

## 프로젝트 개요
- 라즈베리파이5 기반 스마트홈 제어 허브
- NPU(Hailo-8L)에서 제스처 분류, CPU에서 MediaPipe 랜드마크 추출
- Home Assistant와 REST/WebSocket으로 연동

## 하드웨어 환경
- 라즈베리파이5 (aarch64)
- Hailo-8L NPU
- 라즈베리파이 카메라 (9:16)
- 터치스크린 디스플레이

## 제어 대상 기기
- 방조명 (light.room_ceiling) — 손가락 1개(검지)
- 스탠드 조명 (light.standing_lamp) — 손가락 2개(검지, 중지)
- 선풍기 (fan.room_fan) — 손가락 3개(검지, 중지, 약지)
- 에어컨 (climate.aircon) — 손가락 4개(검지, 중지, 약지, 새끼)
- 가습기 (humidifier.room) — 손가락 5개

## 제스처 정의
- 정적: 손가락 수 1~5 → 기기 선택
- 동적: 검지 O궤적=ON, X궤적=OFF, 위스와이프=노브UP, 아래스와이프=노브DOWN

## 추론 파이프라인 아키텍처

### CPU 담당 (변경 없음)
- MediaPipe Hands로 21개 관절 좌표(x, y, z) 추출
- 좌표 전처리 후 NPU로 전달

### NPU 담당 (Hailo-8L, MLP 모델)
- **정적 제스처 모델** (손가락 수 1~5 분류)
  - 입력: `float32 (1, 63)` — 21개 × xyz
  - 출력: `float32 (1, 5)` — logit (클래스 5개)
- **동적 제스처 모델** (O/X/스와이프 분류)
  - 입력: `float32 (1, 1890)` — 시퀀스 30프레임 × 21개 × xyz = 1890
  - 출력: `float32 (1, 4)` — logit (클래스 4개)

### 개발 단계 전략 (Hot Swap)
1. **현재 MVP**: CPU 룰베이스 제스처 인식 (MediaPipe 좌표 → recognizer.py)
2. **파이프라인 검증**: 시중 .pt/.onnx → .hef 변환 후 HailoEngine으로 파이프라인 end-to-end 검증
3. **MLP 모델 교체**: AI팀 개발 MLP .hef만 교체하면 바로 동작 (HailoEngine 코드 변경 없음)
- .hef 파일은 `models/` 디렉토리에 배치, 별도 `cammand-models` 레포에서 관리

## 코드 컨벤션
- Python 3.11+, asyncio 기반
- 타입 힌트 필수
- 한국어 주석 사용

#코드 생성지침
1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

LLMs often pick an interpretation silently and run with it. This principle forces explicit reasoning:

State assumptions explicitly — If uncertain, ask rather than guess
Present multiple interpretations — Don't pick silently when ambiguity exists
Push back when warranted — If a simpler approach exists, say so
Stop when confused — Name what's unclear and ask for clarification
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

Combat the tendency toward overengineering:

No features beyond what was asked
No abstractions for single-use code
No "flexibility" or "configurability" that wasn't requested
No error handling for impossible scenarios
If 200 lines could be 50, rewrite it
The test: Would a senior engineer say this is overcomplicated? If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting
Don't refactor things that aren't broken
Match existing style, even if you'd do it differently
If you notice unrelated dead code, mention it — don't delete it
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused
Don't remove pre-existing dead code unless asked
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform imperative tasks into verifiable goals:

Instead of...	Transform to...
"Add validation"	"Write tests for invalid inputs, then make them pass"
"Fix the bug"	"Write a test that reproduces it, then make it pass"
"Refactor X"	"Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let the LLM loop independently. Weak criteria ("make it work") require constant clarification.