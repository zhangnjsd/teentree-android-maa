use maa_framework::MaaStatus;
use maa_framework::controller::Controller;
use maa_framework::resource::Resource;
use maa_framework::tasker::Tasker;
use maa_framework::toolkit::Toolkit;
use maa_framework::{set_debug_mode, set_stdout_level, sys};
use reqwest::blocking::Client;
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const LLM_API_KEY: &str = "";
const LLM_API_BASE: &str = "http://localhost:11434/api/generate";
const LLM_MODEL: &str = "gpt-oss:120b-cloud";
const LLM_TIMEOUT_SECS: u64 = 30;
const REPEAT_SUBMIT_MIN_STREAK: u8 = 3;
const REPEAT_SUBMIT_MIN_WINDOW_MS: u128 = 1200;

struct RepeatSubmitState {
    last_signature: String,
    last_reco_id: i64,
    streak: u8,
    signature_started_at: Instant,
}

impl RepeatSubmitState {
    fn new() -> Self {
        Self {
            last_signature: String::new(),
            last_reco_id: -1,
            streak: 0,
            signature_started_at: Instant::now(),
        }
    }
}

#[derive(Clone)]
struct OcrEntry {
    y: i64,
    x: i64,
    text: String,
    raw: serde_json::Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QuestionType {
    Single,
    Multi,
    Judge,
    Unknown,
}

impl QuestionType {
    fn as_str(self) -> &'static str {
        match self {
            QuestionType::Single => "single",
            QuestionType::Multi => "multi",
            QuestionType::Judge => "judge",
            QuestionType::Unknown => "unknown",
        }
    }
}

fn collect_ocr_entries(detail: &serde_json::Value) -> Vec<OcrEntry> {
    let mut entries: Vec<OcrEntry> = Vec::new();

    if let Some(arr) = detail.get("all").and_then(|v| v.as_array()) {
        for item in arr {
            let Some(text) = item.get("text").and_then(|v| v.as_str()) else {
                continue;
            };

            let (x, y) = item
                .get("box")
                .and_then(|v| v.as_array())
                .and_then(|b| {
                    let x = b.first()?.as_i64()?;
                    let y = b.get(1)?.as_i64()?;
                    Some((x, y))
                })
                .unwrap_or((i64::MAX, i64::MAX));

            entries.push(OcrEntry {
                y,
                x,
                text: text.to_string(),
                raw: item.clone(),
            });
        }
    }

    entries.sort_by(|a, b| a.y.cmp(&b.y).then(a.x.cmp(&b.x)));
    entries
}

fn clip_quiz_window(entries: &[OcrEntry]) -> Vec<OcrEntry> {
    let start = entries
        .iter()
        .position(|e| e.text.contains("关闭") || e.text.contains("提交作业"));
    let end = entries.iter().position(|e| e.text.contains("下一题"));

    match (start, end) {
        (Some(s), Some(e)) if e >= s => entries[s..=e].to_vec(),
        _ => Vec::new(),
    }
}

fn build_clipped_detail(entries: &[OcrEntry]) -> serde_json::Value {
    let all = entries
        .iter()
        .map(|e| e.raw.clone())
        .collect::<Vec<serde_json::Value>>();

    json!({ "all": all })
}

fn has_quiz_keywords(texts: &[String]) -> bool {
    const QUIZ_KEYWORDS: [&str; 8] = [
        "巩固知识点",
        "不会影响到",
        "上一题",
        "下一题",
        "关闭",
        "判断题",
        "单元测试",
        "选择题选项顺序为随机排序",
    ];
    texts
        .iter()
        .any(|line| QUIZ_KEYWORDS.iter().any(|kw| line.contains(kw)))
}

fn has_correct_result_keywords(texts: &[String]) -> bool {
    texts.iter().any(|line| {
        line.contains("正确答案") || line.contains("回答正确") || line.contains("是最后")
    })
}

fn detect_question_type(texts: &[String]) -> QuestionType {
    if texts.iter().any(|line| line.contains("多选题")) {
        QuestionType::Multi
    } else if texts.iter().any(|line| line.contains("单选题")) {
        QuestionType::Single
    } else if texts.iter().any(|line| line.contains("判断题")) {
        QuestionType::Judge
    } else {
        QuestionType::Unknown
    }
}

fn normalize_answer_keyword(raw: &str) -> String {
    raw.trim_matches(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                ':' | '：' | ',' | '，' | ';' | '；' | '.' | '。' | '"' | '\''
            )
    })
    .to_string()
}

fn normalize_for_match(raw: &str) -> String {
    raw.chars()
        .filter(|c| c.is_ascii_alphanumeric() || ('\u{4e00}'..='\u{9fff}').contains(c))
        .collect::<String>()
}

fn count_common_chars(a: &str, b: &str) -> usize {
    let mut count = 0usize;
    for ch in a.chars() {
        if b.contains(ch) {
            count += 1;
        }
    }
    count
}

fn char_overlap_score(a: &str, b: &str) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let common = count_common_chars(a, b) as f32;
    let shorter = a.chars().count().min(b.chars().count()) as f32;
    if shorter <= 0.0 {
        0.0
    } else {
        common / shorter
    }
}

fn normalize_ocr_option_text(raw: &str) -> String {
    // Keep only useful visible chars and remove frequent OCR separators.
    raw.replace(' ', "")
        .replace('\t', "")
        .replace('　', "")
        .replace("（", "")
        .replace("）", "")
        .replace('(', "")
        .replace(')', "")
}

fn is_option_noise(text: &str) -> bool {
    const NOISE: [&str; 15] = [
        "提交作业",
        "温馨提示",
        "选择题选项顺序",
        "核对答",
        "题卡",
        "上一题",
        "下一题",
        "单元测试",
        "巩固知识点",
        "关闭",
        "单选题",
        "多选题",
        "判断题",
        "正确答案",
        "回答正确",
    ];
    NOISE.iter().any(|kw| text.contains(kw))
}

fn is_pagination_marker(text: &str) -> bool {
    let compact = text.replace(' ', "").replace('　', "");
    if compact.is_empty() {
        return true;
    }

    if compact.contains('/') {
        let mut has_digit = false;
        let all_ok = compact.chars().all(|c| {
            if c.is_ascii_digit() {
                has_digit = true;
                true
            } else {
                c == '/' || c == '-' || c == '|'
            }
        });
        return all_ok && has_digit;
    }

    false
}

fn extract_option_texts(entries: &[OcrEntry]) -> Vec<String> {
    let mut out = Vec::new();
    for e in entries {
        let t = normalize_ocr_option_text(&normalize_answer_keyword(&e.text));
        if t.chars().count() < 1 {
            continue;
        }
        if e.x < 120 {
            continue;
        }
        if is_option_noise(&t) {
            continue;
        }
        if is_pagination_marker(&t) {
            continue;
        }
        if t.contains('？') || t.contains('?') || t.contains('。') {
            continue;
        }
        if !out.iter().any(|x| x == &t) {
            out.push(t);
        }
    }
    out
}

fn derive_option_keyword(option_text: &str) -> String {
    let cleaned = normalize_for_match(option_text);
    let chars = cleaned.chars().collect::<Vec<char>>();
    if chars.is_empty() {
        return String::new();
    }

    if chars.len() <= 6 {
        return chars.iter().collect();
    }

    let start = chars.len().saturating_sub(6);
    let mut tail = chars[start..].iter().collect::<String>();
    while let Some(first) = tail.chars().next() {
        if matches!(first, '的' | '和' | '与' | '及' | '并' | '在') && tail.chars().count() > 2
        {
            tail = tail.chars().skip(1).collect();
        } else {
            break;
        }
    }
    tail
}

fn build_keyword_candidates(option_texts: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for opt in option_texts {
        let kw = derive_option_keyword(opt);
        if kw.chars().count() < 2 {
            continue;
        }
        if !out.iter().any(|x| x == &kw) {
            out.push(kw);
        }
    }
    out
}

fn build_judge_candidates(entries: &[OcrEntry]) -> Vec<String> {
    let mut out = Vec::new();

    for e in entries {
        if e.x < 120 {
            continue;
        }

        let t = normalize_ocr_option_text(&normalize_answer_keyword(&e.text));
        if t.is_empty() {
            continue;
        }
        if is_option_noise(&t) {
            continue;
        }
        if t.contains('？') || t.contains('?') || t.contains('。') {
            continue;
        }
        if t.chars().count() > 10 {
            continue;
        }

        if !out.iter().any(|x| x == &t) {
            out.push(t);
        }
    }

    if out.len() >= 2 {
        return out;
    }

    // OCR 只识别出部分文本时，补充常见判断词，但仍以 OCR 实际内容为准。
    let mut fallback = out;
    let texts = entries
        .iter()
        .map(|e| normalize_ocr_option_text(&e.text))
        .collect::<Vec<String>>();

    let known = ["正确", "错误", "对", "错", "是", "否", "可以", "不可以"];
    for token in known {
        if texts.iter().any(|t| t.contains(token)) && !fallback.iter().any(|x| x == token) {
            fallback.push(token.to_string());
        }
    }

    if fallback.is_empty() {
        vec!["对".to_string(), "错".to_string()]
    } else {
        fallback
    }
}

fn map_to_candidate(answer: &str, candidates: &[String]) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }

    let ans_norm = normalize_for_match(answer);
    if ans_norm.is_empty() {
        return None;
    }

    if let Some(exact) = candidates
        .iter()
        .find(|c| normalize_for_match(c) == ans_norm)
        .cloned()
    {
        return Some(exact);
    }

    if let Some(contained) = candidates
        .iter()
        .filter(|c| ans_norm.contains(&normalize_for_match(c)))
        .max_by_key(|c| c.chars().count())
        .cloned()
    {
        return Some(contained);
    }

    let mut best: Option<(String, f32)> = None;
    for c in candidates {
        let c_norm = normalize_for_match(c);
        if c_norm.is_empty() {
            continue;
        }
        let common = count_common_chars(&ans_norm, &c_norm) as f32;
        let score = common / (c_norm.chars().count() as f32);
        if score < 0.6 {
            continue;
        }
        match &best {
            Some((_, s)) if score <= *s => {}
            _ => best = Some((c.clone(), score)),
        }
    }

    best.map(|(v, _)| v)
}

fn parse_llm_answers(raw: &str, question_type: QuestionType, candidates: &[String]) -> Vec<String> {
    let mut line_answers = raw
        .lines()
        .map(normalize_answer_keyword)
        .filter(|line| !line.is_empty())
        .collect::<Vec<String>>();

    if question_type == QuestionType::Judge {
        let normalized = line_answers
            .drain(..)
            .filter_map(|ans| {
                if let Some(mapped) = map_to_candidate(&ans, candidates) {
                    return Some(mapped);
                }

                let lowered = ans.to_lowercase();
                if (ans.contains("正确") || ans.contains("对") || lowered == "true")
                    && let Some(c) = candidates
                        .iter()
                        .find(|c| c.contains("正确") || c.contains("对") || c == &"是")
                        .cloned()
                {
                    return Some(c);
                }

                if (ans.contains("错误") || ans.contains("错") || lowered == "false")
                    && let Some(c) = candidates
                        .iter()
                        .find(|c| c.contains("错误") || c.contains("错") || c == &"否")
                        .cloned()
                {
                    return Some(c);
                }

                None
            })
            .collect::<Vec<String>>();

        let mut dedup = Vec::new();
        for ans in normalized {
            if !dedup.iter().any(|x| x == &ans) {
                dedup.push(ans);
            }
        }

        if dedup.len() == 1 {
            return vec![dedup[0].clone()];
        }
        return Vec::new();
    }

    let mut dedup = Vec::new();
    for ans in line_answers.drain(..) {
        if let Some(mapped) = map_to_candidate(&ans, candidates) {
            if !dedup.iter().any(|x| x == &mapped) {
                dedup.push(mapped);
            }
        }
    }

    match question_type {
        QuestionType::Multi => {
            if dedup.len() >= 2 {
                return dedup;
            }

            // Multi-choice must not be single-line output.
            let fallback = raw
                .split(|c| matches!(c, '、' | ',' | '，' | ';' | '；' | '|' | '/' | '\\'))
                .map(normalize_answer_keyword)
                .filter(|line| !line.is_empty())
                .collect::<Vec<String>>();

            let mut dedup_fallback = Vec::new();
            for ans in fallback {
                if let Some(mapped) = map_to_candidate(&ans, candidates) {
                    if !dedup_fallback.iter().any(|x| x == &mapped) {
                        dedup_fallback.push(mapped);
                    }
                }
            }

            if dedup_fallback.len() >= 2 {
                dedup_fallback
            } else {
                Vec::new()
            }
        }
        QuestionType::Single | QuestionType::Judge => {
            // Single/Judge must be exactly one answer line.
            if dedup.len() == 1 {
                vec![dedup[0].clone()]
            } else {
                Vec::new()
            }
        }
        QuestionType::Unknown => {
            if dedup.len() == 1 {
                vec![dedup[0].clone()]
            } else {
                Vec::new()
            }
        }
    }
}

fn box_center(raw: &serde_json::Value) -> Option<(i32, i32)> {
    let arr = raw.get("box")?.as_array()?;
    let x = arr.first()?.as_i64()?;
    let y = arr.get(1)?.as_i64()?;
    let w = arr.get(2)?.as_i64()?;
    let h = arr.get(3)?.as_i64()?;

    Some(((x + w / 2) as i32, (y + h / 2) as i32))
}

fn find_click_point(entries: &[OcrEntry], text: &str) -> Option<(i32, i32)> {
    let exact = entries
        .iter()
        .find(|e| e.text == text)
        .and_then(|e| box_center(&e.raw))
        .or_else(|| {
            entries
                .iter()
                .find(|e| e.text.contains(text))
                .and_then(|e| box_center(&e.raw))
        });

    if exact.is_some() {
        return exact;
    }

    let target = normalize_for_match(text);
    if target.is_empty() {
        return None;
    }

    if let Some(p) = entries.iter().find_map(|e| {
        let t = normalize_for_match(&e.text);
        if t.is_empty() {
            return None;
        }
        if t.contains(&target) || target.contains(&t) {
            box_center(&e.raw)
        } else {
            None
        }
    }) {
        return Some(p);
    }

    // Fuzzy fallback for OCR split/garbled fragments.
    entries
        .iter()
        .filter_map(|e| {
            let t = normalize_for_match(&e.text);
            if t.is_empty() {
                return None;
            }
            let score = char_overlap_score(&target, &t);
            if score >= 0.62 {
                box_center(&e.raw).map(|p| (p, score))
            } else {
                None
            }
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(p, _)| p)
}

fn parse_progress_pair(s: &str) -> Option<(i32, i32)> {
    let chars: Vec<char> = s.chars().collect();
    for (idx, ch) in chars.iter().enumerate() {
        if *ch != '/' && *ch != '／' {
            continue;
        }

        let mut left_end = idx;
        while left_end > 0 && chars[left_end - 1].is_whitespace() {
            left_end -= 1;
        }
        let mut left_start = left_end;
        while left_start > 0 && chars[left_start - 1].is_ascii_digit() {
            left_start -= 1;
        }

        let mut right_start = idx + 1;
        while right_start < chars.len() && chars[right_start].is_whitespace() {
            right_start += 1;
        }
        let mut right_end = right_start;
        while right_end < chars.len() && chars[right_end].is_ascii_digit() {
            right_end += 1;
        }

        if left_start == left_end || right_start == right_end {
            continue;
        }

        let left: String = chars[left_start..left_end].iter().collect();
        let right: String = chars[right_start..right_end].iter().collect();

        let current = left.parse::<i32>().ok()?;
        let total = right.parse::<i32>().ok()?;
        if current <= 0 || total <= 0 || current > total {
            continue;
        }

        return Some((current, total));
    }

    None
}

fn detect_progress(entries: &[OcrEntry]) -> Option<(i32, i32)> {
    entries.iter().find_map(|e| parse_progress_pair(&e.text))
}

fn call_llm(prompt: &str, user_content: &str) -> Result<String, String> {
    let api_key = LLM_API_KEY;
    let api_base = LLM_API_BASE;
    let model = LLM_MODEL;
    let timeout_secs = LLM_TIMEOUT_SECS;

    let client = Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| format!("LLM HTTP client 初始化失败: {e}"))?;

    let use_generate_api = api_base.contains("/api/generate");
    let merged_prompt = format!("{}\n\n{}", prompt, user_content);

    let body = if use_generate_api {
        json!({
            "model": model,
            "prompt": merged_prompt,
            "stream": false,
            "options": { "temperature": 0.16 }
        })
    } else {
        json!({
            "model": model,
            "messages": [
                { "role": "system", "content": format!("{}", prompt) },
                { "role": "user", "content": format!("{}", user_content) }
            ],
            "stream": false,
            "temperature": 0.16
        })
    };

    for attempt in 0..2 {
        let mut request = client.post(api_base).json(&body);
        if !api_key.trim().is_empty() {
            request = request.bearer_auth(api_key);
        }

        let resp = request.send().map_err(|e| format!("请求 LLM 失败: {e}"))?;
        let status = resp.status();
        let resp_text = resp.text().map_err(|e| format!("读取 LLM 响应失败: {e}"))?;

        if !status.is_success() {
            return Err(format!("LLM 返回非 2xx: {} | {}", status, resp_text));
        }

        let value: serde_json::Value = serde_json::from_str(&resp_text)
            .map_err(|e| format!("解析 LLM 响应 JSON 失败: {e}"))?;

        let chat_content = value
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("message"))
            .and_then(|v| v.get("content"))
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        if let Some(content) = chat_content {
            return Ok(content);
        }

        let gen_content = value
            .get("response")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        if let Some(content) = gen_content {
            return Ok(content);
        }

        let is_model_loading = value
            .get("done_reason")
            .and_then(|v| v.as_str())
            .map(|s| s == "load")
            .unwrap_or(false);

        if is_model_loading && attempt == 0 {
            thread::sleep(Duration::from_millis(600));
            continue;
        }

        return Err(format!("LLM 响应缺少有效内容: {}", resp_text));
    }

    Err("LLM 请求失败: 重试后仍无有效内容".to_string())
}

fn try_click_confirm(entries: &[OcrEntry], controller: &Controller) -> bool {
    if let Some((x, y)) = find_click_point(entries, "确认") {
        match controller.post_click(x, y) {
            Ok(job) => {
                let status = controller.wait(job);
                println!("[QUIZ_CONFIRM_CLICK] ({}, {}) | status={:?}", x, y, status);
                true
            }
            Err(e) => {
                eprintln!("[QUIZ_CONFIRM_CLICK_ERROR] {:?}", e);
                false
            }
        }
    } else {
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Toolkit::init_option("./", "{}")?;
    // Turn on verbose framework logs so node-level debug details can be printed.
    set_debug_mode(true)?;
    set_stdout_level(sys::MaaLoggingLevelEnum_MaaLoggingLevel_All as i32)?;

    let devices = Toolkit::find_adb_devices()?;
    if devices.is_empty() {
        eprintln!("No ADB device found");
        return Ok(());
    }

    let device = &devices[0];
    let controller = Controller::new_adb(
        device.adb_path.to_str().unwrap(),
        &device.address,
        &device.config.to_string(),
        "",
    )?;
    let connect_job = controller.post_connection()?;
    let connect_status = controller.wait(connect_job);
    if connect_status != MaaStatus::SUCCEEDED || !controller.connected() {
        eprintln!("Failed to connect controller, status: {:?}", connect_status);
        return Ok(());
    }

    let resource = Resource::new()?;
    let bundle_job = resource.post_bundle("./resource")?;
    let bundle_status = bundle_job.wait();
    if bundle_status != MaaStatus::SUCCEEDED || !resource.loaded() {
        eprintln!(
            "Failed to load resource bundle, status: {:?}",
            bundle_status
        );
        return Ok(());
    }

    let tasker = Tasker::new()?;
    tasker.bind_controller(&controller)?;
    tasker.bind_resource(&resource)?;

    tasker.add_sink(|msg, detail| {
        println!("[TASK] {msg} | {detail}");
    })?;
    tasker.add_context_sink(|msg, detail| {
        println!("[CTX ] {msg} | {detail}");
    })?;
    let action_controller = controller.clone();
    let llm_inflight = Arc::new(AtomicBool::new(false));
    let llm_inflight_guard = Arc::clone(&llm_inflight);
    let repeat_submit_state = Arc::new(Mutex::new(RepeatSubmitState::new()));
    let repeat_submit_state_guard = Arc::clone(&repeat_submit_state);
    let submit_confirm_pending = Arc::new(Mutex::new(0u8));
    let submit_confirm_pending_guard = Arc::clone(&submit_confirm_pending);
    tasker.add_context_sink(move |msg, detail| {
        if msg != maa_framework::notification::msg::NODE_RECOGNITION_SUCCEEDED
            && msg != maa_framework::notification::msg::NODE_RECOGNITION_FAILED
        {
            return;
        }

        if llm_inflight_guard.load(Ordering::Acquire) {
            return;
        }

        let payload: serde_json::Value = match serde_json::from_str(detail) {
            Ok(v) => v,
            Err(_) => return,
        };

        let node_name = payload.get("name").and_then(|v| v.as_str()).unwrap_or("");

        let reco_id = match payload.get("reco_id").and_then(|v| v.as_i64()) {
            Some(v) => v,
            None => return,
        };

        let reco_details = payload.get("reco_details").cloned().unwrap_or_default();
        let detail_obj = reco_details
            .get("detail")
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        let entries = collect_ocr_entries(&detail_obj);
        if entries.is_empty() {
            return;
        }

        // Post-submit confirm handling: for a few subsequent OCR rounds,
        // try to click "确认" if it appears; otherwise ignore when budget ends.
        if let Ok(mut pending) = submit_confirm_pending_guard.lock()
            && *pending > 0
        {
            if try_click_confirm(&entries, &action_controller) {
                *pending = 0;
            } else {
                *pending = pending.saturating_sub(1);
                if *pending == 0 {
                    println!("[QUIZ_CONFIRM_SKIP] 未识别到确认，已忽略");
                }
            }
        }

        let entry_texts = entries
            .iter()
            .map(|e| e.text.clone())
            .collect::<Vec<String>>();

        if has_correct_result_keywords(&entry_texts) {
            if let Some((x, y)) = find_click_point(&entries, "关闭") {
                match action_controller.post_click(x, y) {
                    Ok(job) => {
                        let close_status = action_controller.wait(job);
                        println!(
                            "[QUIZ_CLOSE_BY_RESULT] ({}, {}) | status={:?}",
                            x, y, close_status
                        );
                    }
                    Err(e) => {
                        eprintln!("[QUIZ_CLOSE_BY_RESULT_ERROR] {:?}", e);
                    }
                }
            } else {
                eprintln!("[QUIZ_CLOSE_BY_RESULT_MISS] 识别到答题结果但未找到关闭按钮");
            }
            return;
        }

        let clipped_entries = clip_quiz_window(&entries);
        if clipped_entries.is_empty() {
            return;
        }

        let clipped_texts = clipped_entries
            .iter()
            .map(|e| e.text.clone())
            .collect::<Vec<String>>();

        if !has_quiz_keywords(&clipped_texts) {
            return;
        }

        let clipped_detail = build_clipped_detail(&clipped_entries);
        let question_type = detect_question_type(&clipped_texts);
        let option_texts = if question_type == QuestionType::Judge {
            Vec::new()
        } else {
            extract_option_texts(&clipped_entries)
        };
        let keyword_candidates = if question_type == QuestionType::Judge {
            build_judge_candidates(&clipped_entries)
        } else {
            build_keyword_candidates(&option_texts)
        };

        let current_signature = format!(
            "{}|{}|{}|{}",
            question_type.as_str(),
            clipped_texts
                .iter()
                .map(|s| normalize_for_match(s))
                .collect::<Vec<String>>()
                .join("||"),
            option_texts
                .iter()
                .map(|s| normalize_for_match(s))
                .collect::<Vec<String>>()
                .join("||"),
            keyword_candidates
                .iter()
                .map(|s| normalize_for_match(s))
                .collect::<Vec<String>>()
                .join("||")
        );

        let should_direct_submit_repeat = if let Ok(mut st) = repeat_submit_state_guard.lock() {
            let now = Instant::now();
            let same_signature = st.last_signature == current_signature;
            let reco_advanced = st.last_reco_id != reco_id;

            if same_signature && reco_advanced {
                st.streak = st.streak.saturating_add(1);
            } else if !same_signature {
                st.streak = 1;
                st.signature_started_at = now;
            }

            let elapsed_ms = now.duration_since(st.signature_started_at).as_millis();
            let ready = same_signature
                && reco_advanced
                && st.streak >= REPEAT_SUBMIT_MIN_STREAK
                && elapsed_ms >= REPEAT_SUBMIT_MIN_WINDOW_MS;

            st.last_signature = current_signature;
            st.last_reco_id = reco_id;
            ready
        } else {
            false
        };

        if should_direct_submit_repeat {
            if let Some((x, y)) = find_click_point(&clipped_entries, "提交作业") {
                match action_controller.post_click(x, y) {
                    Ok(job) => {
                        let submit_status = action_controller.wait(job);
                        println!(
                            "[QUIZ_DIRECT_SUBMIT_REPEAT] ({}, {}) | status={:?}",
                            x, y, submit_status
                        );
                        if !try_click_confirm(&entries, &action_controller) {
                            if let Ok(mut pending) = submit_confirm_pending_guard.lock() {
                                *pending = 3;
                            }
                            println!("[QUIZ_CONFIRM_PENDING] 等待后续OCR识别确认");
                        }
                    }
                    Err(e) => {
                        eprintln!("[QUIZ_DIRECT_SUBMIT_REPEAT_ERROR] {:?}", e);
                    }
                }
            } else {
                eprintln!("[QUIZ_DIRECT_SUBMIT_REPEAT_MISS] 未找到提交作业按钮");
            }
            return;
        }

        if question_type != QuestionType::Judge && keyword_candidates.is_empty() {
            eprintln!("[KEYWORD_CANDIDATES_EMPTY] 未提取到有效选项关键词");
            return;
        }

        let llm_payload = json!({
            "type": "quiz_ocr",
            "event": msg,
            "node": node_name,
            "reco_id": reco_id,
            "question_type": question_type.as_str(),
            "option_texts": option_texts,
            "keyword_candidates": keyword_candidates,
            "texts": clipped_texts,
            "raw_detail": clipped_detail,
        });

        println!("[LLM_INPUT] {}", llm_payload);

        // Avoid overlapping OCR->LLM requests: while one question is being solved,
        // ignore subsequent recognition callbacks until the current request returns.
        if llm_inflight_guard
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        const PROMPT: &str = "你是一个智能学习助手，正在帮助学生解答一个题目。
以下是从题目界面OCR识别得到的文本内容，包含题干和选项。
输入 JSON 中有 question_type 字段，取值为 single / multi / judge / unknown。
输入 JSON 中还有 option_texts 和 keyword_candidates。
你可以基于 option_texts 对 OCR 进行纠错理解，但最终输出必须严格从 keyword_candidates 中选择。
每个正确答案只能输出一个关键词（即 keyword_candidates 里的一个元素），禁止输出完整选项句子。
你必须严格遵守 question_type 的输出格式约束，不允许越界。
不要输出ABCD等选项标签，也不要输出题干内容和其他提示性内容。
多选题只有最多四个选项，请根据识别合理组合选项内容，不要对一个选项返回多个关键词！
这样意味着你**至多**返回**四个**关键词，其应当对应**四个**选项！
判断题也要在`parsed`字段只输出一个关键词!
有些多选题的选项较长，请合理组合选项内容并提炼关键词，不要对一个选项返回多个关键词，也不要将一个选项拆分为多个选项！
选项可能有B、C、D等标签，但这些标签不可信，不要对标签依赖，因为多行选项标签会把题目选项拆分。
若 question_type=single 或 judge：只能输出1行，且只能有1个答案关键词。
若 question_type=multi：必须输出2行及以上，每行1个答案关键词。
若 question_type=unknown：按 single 处理，只能输出1行。
输出后立刻结束，不要解释。";

        const RETRY_PROMPT: &str = "重新严格输出。
你上次输出不符合格式。
必须只输出答案关键词，不要解释。
single/judge/unknown: 只能1行。
multi: 必须2行及以上，每行1个答案。";

        match call_llm(PROMPT, &llm_payload.to_string()) {
            Ok(answer) => {
                let mut answers = parse_llm_answers(&answer, question_type, &keyword_candidates);
                println!(
                    "[LLM_OUTPUT] question_type={} | raw={} | parsed={:?}",
                    question_type.as_str(), answer, answers
                );

                if answers.is_empty() {
                    match call_llm(RETRY_PROMPT, &llm_payload.to_string()) {
                        Ok(retry_answer) => {
                            answers = parse_llm_answers(&retry_answer, question_type, &keyword_candidates);
                            println!(
                                "[LLM_RETRY_OUTPUT] question_type={} | raw={} | parsed={:?}",
                                question_type.as_str(), retry_answer, answers
                            );
                        }
                        Err(e) => {
                            eprintln!("[LLM_RETRY_ERROR] {}", e);
                        }
                    }
                }

                if answers.is_empty() {
                    eprintln!(
                        "[ANSWER_PARSE_EMPTY] LLM 返回内容不符合题型格式约束: question_type={}",
                        question_type.as_str()
                    );
                }

                for ans in answers {
                    if let Some((x, y)) = find_click_point(&clipped_entries, &ans) {
                        match action_controller.post_click(x, y) {
                            Ok(job) => {
                                let click_status = action_controller.wait(job);
                                println!(
                                    "[ANSWER_CLICK] {} -> ({}, {}) | status={:?}",
                                    ans, x, y, click_status
                                );
                            }
                            Err(e) => {
                                eprintln!("[ANSWER_CLICK_ERROR] {} | ({}, {}) | {:?}", ans, x, y, e);
                            }
                        }
                    } else {
                        eprintln!("[ANSWER_CLICK_MISS] 未找到与答案匹配的OCR项: {}", ans);
                    }

                    thread::sleep(Duration::from_millis(120));
                }

                thread::sleep(Duration::from_millis(250));

                let progress = detect_progress(&entries);
                let nav_label = "下一题";

                if let Some((x, y)) = find_click_point(&clipped_entries, nav_label) {
                    match action_controller.post_click(x, y) {
                        Ok(job) => {
                            let nav_status = action_controller.wait(job);
                            println!(
                                "[QUIZ_NAV] {} -> ({}, {}) | progress={:?} | status={:?}",
                                nav_label, x, y, progress, nav_status
                            );
                        }
                        Err(e) => {
                            eprintln!("[QUIZ_NAV_ERROR] {} | {:?}", nav_label, e);
                        }
                    }
                } else {
                    eprintln!(
                        "[QUIZ_NAV_MISS] 未找到按钮: {} | progress={:?}",
                        nav_label, progress
                    );
                }
            }
            Err(e) => {
                eprintln!("[LLM_ERROR] {}", e);
            }
        }

        llm_inflight_guard.store(false, Ordering::Release);
    })?;

    if !tasker.inited() {
        eprintln!("Failed to initialize MAA");
        return Ok(());
    }

    let task_job = tasker.post_task("Startup", "{}")?;
    println!("Task started!");
    let task_status = task_job.wait();
    println!("Task finished with status: {:?}", task_status);

    if let Some(task_detail) = task_job.get(false)? {
        println!("Task detail entry: {}", task_detail.entry);
        for node in task_detail.nodes.into_iter().flatten() {
            println!(
                "[NODE] {} | completed={} | reco_id={} | act_id={}",
                node.node_name, node.completed, node.reco_id, node.act_id
            );
        }
    }

    Ok(())
}
