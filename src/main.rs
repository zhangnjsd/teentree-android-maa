use maa_framework::MaaStatus;
use maa_framework::controller::Controller;
use maa_framework::resource::Resource;
use maa_framework::tasker::Tasker;
use maa_framework::toolkit::Toolkit;
use maa_framework::{set_debug_mode, set_stdout_level, sys};
use reqwest::blocking::Client;
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

const LLM_API_KEY: &str = "";
const LLM_API_BASE: &str = "http://localhost:11434/api/generate";
const LLM_MODEL: &str = "deepseek-v3.1:671b-cloud";
const LLM_TIMEOUT_SECS: u64 = 30;

#[derive(Clone)]
struct OcrEntry {
    y: i64,
    x: i64,
    text: String,
    raw: serde_json::Value,
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
    let start = entries.iter().position(|e| e.text.contains("关闭"));
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
    const QUIZ_KEYWORDS: [&str; 6] = [
        "巩固知识点",
        "不会影响到",
        "上一题",
        "下一题",
        "关闭",
        "判断题",
    ];
    texts
        .iter()
        .any(|line| QUIZ_KEYWORDS.iter().any(|kw| line.contains(kw)))
}

fn has_correct_result_keywords(texts: &[String]) -> bool {
    texts
        .iter()
        .any(|line| line.contains("正确答案") || line.contains("回答正确"))
}

fn is_multi_choice_question(texts: &[String]) -> bool {
    texts.iter().any(|line| line.contains("多选题"))
}

fn normalize_answer_keyword(raw: &str) -> String {
    raw
        .trim_matches(|c: char| {
            c.is_whitespace()
                || matches!(
                    c,
                    ':' | '：' | ',' | '，' | ';' | '；' | '.' | '。' | '"' | '\''
                )
        })
        .to_string()
}

fn parse_llm_answers(raw: &str, is_multi: bool) -> Vec<String> {
    if is_multi {
        let mut answers = raw
            .lines()
            .map(normalize_answer_keyword)
            .filter(|line| !line.is_empty())
            .collect::<Vec<String>>();

        // Some models still return one line joined by separators for multi-choice.
        if answers.len() <= 1 {
            let fallback = raw
                .split(|c| matches!(c, '、' | ',' | '，' | ';' | '；' | '|' | '/' | '\\'))
                .map(normalize_answer_keyword)
                .filter(|line| !line.is_empty())
                .collect::<Vec<String>>();
            if !fallback.is_empty() {
                answers = fallback;
            }
        }

        let mut dedup = Vec::new();
        for ans in answers {
            if !dedup.iter().any(|x| x == &ans) {
                dedup.push(ans);
            }
        }
        dedup
    } else {
        raw.lines()
            .map(normalize_answer_keyword)
            .find(|line| !line.is_empty())
            .map(|line| vec![line])
            .unwrap_or_default()
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
    entries
        .iter()
        .find(|e| e.text == text)
        .and_then(|e| box_center(&e.raw))
        .or_else(|| {
            entries
                .iter()
                .find(|e| e.text.contains(text))
                .and_then(|e| box_center(&e.raw))
        })
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

        let llm_payload = json!({
            "type": "quiz_ocr",
            "event": msg,
            "node": node_name,
            "reco_id": reco_id,
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
请分析这些文本，判断这道题的类型（选择题、判断题等），并提取出题干和选项。
输出请**只给出上面的文本内容所包含的选项独有的关键字**，
不要输出ABCD等选项标签，也不要输出题干内容和其他提示性内容。
如果识别到“多选题”，可以输出多个关键字，必须一行一个关键字。
如果不是多选题，只输出1个关键字。
输出后立刻结束，不要解释。";

        let is_multi = is_multi_choice_question(&clipped_texts);

        match call_llm(PROMPT, &llm_payload.to_string()) {
            Ok(answer) => {
                let answers = parse_llm_answers(&answer, is_multi);
                println!(
                    "[LLM_OUTPUT] is_multi={} | raw={} | parsed={:?}",
                    is_multi, answer, answers
                );

                if answers.is_empty() {
                    eprintln!("[ANSWER_PARSE_EMPTY] LLM 返回内容无法解析为可点击答案");
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
