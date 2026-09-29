def predict_answer( #Alternative predict_answer def, use it to get noslide results, unusable with phobert therefore originally discarded in favor of universal sliding windown inference method that phobert uses
    question,
    context,
    debug=False
):


    max_input_length = 512

    # --------------------------------------------------------
    # Tokenize entire question/context pair
    # --------------------------------------------------------

    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_input_length,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # --------------------------------------------------------
    # Move tensors to device
    # --------------------------------------------------------

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    if debug:
        print(
            "Input shape:",
            inputs["input_ids"].shape
        )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # --------------------------------------------------------
    # Get token sequence IDs
    #
    # 0 = question
    # 1 = context
    # None = special tokens
    # --------------------------------------------------------

    sequence_ids = inputs["input_ids"].new_tensor(
        [
            -1 if x is None else x
            for x in tokenizer(
                question,
                context,
                truncation="only_second",
                max_length=max_input_length,
            ).sequence_ids()
        ]
    ).to(device)

    context_positions = (
        sequence_ids == 1
    ).nonzero(as_tuple=True)[0]

    if len(context_positions) == 0:
        return "", 0.0

    context_start = context_positions[0].item()
    context_end = context_positions[-1].item()

    # --------------------------------------------------------
    # Only search inside context
    # --------------------------------------------------------

    context_start_logits = start_logits[
        context_start:
        context_end + 1
    ]

    context_end_logits = end_logits[
        context_start:
        context_end + 1
    ]

    # --------------------------------------------------------
    # Candidate positions
    # --------------------------------------------------------

    top_k = min(
        20,
        len(context_positions)
    )

    start_values, start_indices = torch.topk(
        context_start_logits,
        top_k
    )

    end_values, end_indices = torch.topk(
        context_end_logits,
        top_k
    )

    best_answer = ""
    best_score = float("-inf")

    max_answer_length = 64

    for start_value, start_local in zip(
        start_values,
        start_indices
    ):

        for end_value, end_local in zip(
            end_values,
            end_indices
        ):

            start_local = int(start_local)
            end_local = int(end_local)

            if end_local < start_local:
                continue

            if (
                end_local
                - start_local
                + 1
                > max_answer_length
            ):
                continue

            start_position = (
                context_start
                + start_local
            )

            end_position = (
                context_start
                + end_local
            )

            answer_ids = inputs["input_ids"][
                0,
                start_position:
                end_position + 1
            ]

            answer = tokenizer.decode(
                answer_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

            if not answer:
                continue

            score = (
                start_value
                + end_value
            ).item()

            if score > best_score:

                best_score = score
                best_answer = answer

    if best_score == float("-inf"):
        return "", 0.0

    return best_answer, best_score