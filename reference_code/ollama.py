# Python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "E:/ai_model/Qwen/Qwen3-4B"  # 사용자 환경에 맞춰 수정

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

chat_history = []

print("Qwen3 Chat 시작 (종료하려면 'exit' 입력)")
print("※ '/think'를 포함하면 사고 모드, 포함하지 않으면 비사고 모드로 작동합니다.")

while True:
    user_input = input("\nUser: ").strip()
    if user_input.lower() == "exit":
        print("채팅을 종료합니다.")
        break

    # 프롬프트에 /think 포함 여부 확인
    thinking_enabled = "/think" in user_input
    cleaned_input = user_input.replace("/think", "").replace("/no_think", "").strip()

    chat_history.append({"role": "user", "content": cleaned_input})

    # chat 템플릿 생성
    prompt_text = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # 항상 True로 유지, soft switch는 프롬프트로 판단됨
    )

    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    # 사고 모드와 비사고 모드에 따른 생성 파라미터 설정
    if thinking_enabled:
        temperature = 0.6
        top_p = 0.95
        top_k = 20
    else:
        temperature = 0.7
        top_p = 0.8
        top_k = 20

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_new_tokens=1
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # 사고 모드 결과 추출
    try:
        end_token_id = tokenizer.convert_tokens_to_ids("</think>")
        index = len(output_ids) - output_ids[::-1].index(end_token_id)
    except ValueError:
        index = 0

    thinking_output = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    response_output = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

    if thinking_enabled and thinking_output:
        print(f"\n[Thinking]: {thinking_output}")
    print(f"\nQwen3: {response_output}")

    chat_history.append({"role": "assistant", "content": response_output})