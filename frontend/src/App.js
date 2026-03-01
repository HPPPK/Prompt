import React, { useState, useRef, useEffect, useCallback } from 'react';
import InputBox from './components/InputBox';
import ModelCard from './components/ModelCard';
import CompareView from './components/CompareView';
import HistoryPanel from './components/HistoryPanel';
import { analyzeInput, fetchHistory, checkHealth } from './api';
import './App.css';

const TASK_TYPE_LABELS = {
  writing: '✍️ 写作',
  coding: '💻 编程',
  academic: '📚 学术',
  business: '📊 商业',
  search: '🔎 搜索',
  reasoning: '🧩 推理',
};

const COMPLEXITY_LABELS = { low: '简单', medium: '中等', high: '复杂' };

const EXAMPLES = [
  { text: '帮我分析特斯拉商业模式', icon: '📊' },
  { text: '帮我写一篇小红书文案', icon: '✍️' },
  { text: '用Python写一个爬虫', icon: '💻' },
  { text: '帮我写一篇文献综述', icon: '📚' },
  { text: '证明根号2是无理数', icon: '🧩' },
  { text: '对比iPhone和Pixel哪个好', icon: '🔎' },
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [regeneratingIndex, setRegeneratingIndex] = useState(null);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [darkMode, setDarkMode] = useState(
    () => localStorage.getItem('darkMode') === 'true'
  );
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState([]);
  const [ollamaStatus, setOllamaStatus] = useState(null);
  const [viewMode, setViewMode] = useState('cards'); // 'cards' | 'compare'
  const messagesEndRef = useRef(null);

  // Dark mode
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('darkMode', darkMode);
  }, [darkMode]);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Health check on mount
  useEffect(() => {
    checkHealth().then(setOllamaStatus);
  }, []);

  const loadHistory = useCallback(async () => {
    const data = await fetchHistory();
    setHistory(data.history || []);
  }, []);

  const handleSubmit = async (input) => {
    setMessages((prev) => [...prev, { type: 'user', text: input }]);
    setLoading(true);

    try {
      const data = await analyzeInput(input);
      setMessages((prev) => [...prev, { type: 'result', data }]);
      loadHistory();
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { type: 'error', text: err.message || '分析失败，请重试' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleHistorySelect = (input) => {
    setShowHistory(false);
    handleSubmit(input);
  };

  const toggleHistory = () => {
    if (!showHistory) loadHistory();
    setShowHistory(!showHistory);
  };

  const findPreviousUserInput = useCallback((fromIndex) => {
    for (let idx = fromIndex - 1; idx >= 0; idx -= 1) {
      if (messages[idx]?.type === 'user') return messages[idx].text;
    }
    return '';
  }, [messages]);

  const handleRegenerate = useCallback(async (messageIndex) => {
    if (loading || regeneratingIndex !== null) return;

    const msg = messages[messageIndex];
    const originalInput =
      msg?.type === 'result'
        ? (msg.data?.input || '')
        : findPreviousUserInput(messageIndex);

    if (!originalInput) return;

    setRegeneratingIndex(messageIndex);
    try {
      const data = await analyzeInput(originalInput);
      setMessages((prev) =>
        prev.map((item, idx) => (idx === messageIndex ? { type: 'result', data } : item))
      );
      loadHistory();
    } catch (err) {
      setMessages((prev) =>
        prev.map((item, idx) =>
          idx === messageIndex
            ? { type: 'error', text: err.message || '重新生成失败，请重试' }
            : item
        )
      );
    } finally {
      setRegeneratingIndex(null);
    }
  }, [findPreviousUserInput, loading, messages, regeneratingIndex, loadHistory]);

  const handleCopyResult = useCallback(async (messageIndex) => {
    const msg = messages[messageIndex];
    if (!msg || msg.type !== 'result') return;

    const data = msg.data || {};
    const text = (data.recommendations || [])
      .map((rec) => `=== ${rec.model.name} ===\n推荐理由：${rec.reason}\n\n${rec.prompt}`)
      .join('\n\n');

    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    setCopiedIndex(messageIndex);
    setTimeout(() => setCopiedIndex(null), 1500);
  }, [messages]);

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-top">
          <button className="icon-btn" onClick={toggleHistory} title="历史记录">
            &#x1F553;
          </button>
          <div className="header-center">
            <h1>AI提示词管家</h1>
            <p>智能推荐最佳AI模型 &middot; 生成专属高质量提示词</p>
          </div>
          <button
            className="icon-btn"
            onClick={() => setDarkMode(!darkMode)}
            title={darkMode ? '浅色模式' : '深色模式'}
          >
            {darkMode ? '☀️' : '🌙'}
          </button>
        </div>
        {ollamaStatus && (
          <div className="status-bar">
            <span className={`status-dot ${ollamaStatus.ollama_available ? 'online' : 'offline'}`} />
                    <span>
              {ollamaStatus.ollama_available
                ? `LLM智能模式 (${ollamaStatus.ollama_model})`
                : '基础模式（配置 Gemini API Key 可解锁 LLM 智能分析）'}
            </span>
                  </div>
                )}
      </header>

      <div className="app-body">
        {/* History sidebar */}
        {showHistory && (
          <HistoryPanel
            history={history}
            onSelect={handleHistorySelect}
            onClose={() => setShowHistory(false)}
          />
        )}

        {/* Main content */}
        <main className="messages">
          {messages.length === 0 && (
            <div className="welcome">
              <div className="welcome-icon">✨</div>
              <h2>欢迎使用 AI提示词管家</h2>
              <p>告诉我你想让AI做什么，我会为你推荐最合适的模型和提示词</p>
              <div className="examples">
                <p>试试这些：</p>
                <div className="example-chips">
                  {EXAMPLES.map((ex) => (
                    <button
                      key={ex.text}
                      className="example-chip"
                      onClick={() => handleSubmit(ex.text)}
                      disabled={loading}
                    >
                      <span className="chip-icon">{ex.icon}</span>
                      {ex.text}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {messages.map((msg, i) => {
            if (msg.type === 'user') {
              return (
                <div key={i} className="message user-message">
                  <div className="message-bubble">{msg.text}</div>
                </div>
              );
            }
            if (msg.type === 'error') {
              return (
                <div key={i} className="message error-message">
                  <div className="message-bubble">{msg.text}</div>
                  <div className="message-actions">
                    <button
                      className="action-btn"
                      onClick={() => handleRegenerate(i)}
                      disabled={loading || regeneratingIndex !== null}
                    >
                      {regeneratingIndex === i ? '重试中...' : '重试'}
                    </button>
                  </div>
                </div>
              );
            }

            const { data } = msg;
            const cls = data.classification || {};

            return (
              <div key={i} className="message system-message fade-in">
                {/* Classification info */}
                <div className="classification-info">
                  <div className="task-types">
                    <span>任务类型：</span>
                    {cls.task_types?.map((t, j) => (
                      <span key={j} className="task-type-tag">
                        {TASK_TYPE_LABELS[t.type] || t.type}
                        {t.confidence != null && (
                          <span className="confidence">
                            {Math.round(t.confidence * 100)}%
                          </span>
                        )}
                      </span>
                    ))}
                  </div>
                  <div className="classification-meta">
                    {cls.complexity && (
                      <span className="meta-tag complexity">
                        复杂度: {COMPLEXITY_LABELS[cls.complexity] || cls.complexity}
                      </span>
                    )}
                    {cls.intent && cls.intent !== data.input && (
                      <span className="meta-tag intent" title={cls.intent}>
                        意图: {cls.intent.length > 30 ? cls.intent.slice(0, 30) + '...' : cls.intent}
                      </span>
                    )}
                    {cls.source && (
                      <span className={`meta-tag source-${cls.source}`}>
                        {cls.source === 'llm' ? 'LLM分析' : '关键词匹配'}
                      </span>
                    )}
                    {data.meta?.elapsed_seconds != null && (
                      <span className="meta-tag time">
                        {data.meta.elapsed_seconds}s
                      </span>
                    )}
                  </div>
                </div>

                {/* View mode toggle */}
                {data.recommendations?.length >= 2 && (
                  <div className="view-toggle">
                    <button
                      className={viewMode === 'cards' ? 'active' : ''}
                      onClick={() => setViewMode('cards')}
                    >
                      卡片视图
                    </button>
                    <button
                      className={viewMode === 'compare' ? 'active' : ''}
                      onClick={() => setViewMode('compare')}
                    >
                      对比视图
                    </button>
                  </div>
                )}

                {/* Cards or Compare view */}
                {viewMode === 'compare' && data.recommendations?.length >= 2 ? (
                  <CompareView recommendations={data.recommendations} />
                ) : (
                  <div className="cards">
                    {data.recommendations?.map((rec, j) => (
                      <ModelCard key={j} recommendation={rec} index={j} />
                    ))}
                  </div>
                )}
                <div className="message-actions">
                  <button
                    className="action-btn"
                    onClick={() => handleRegenerate(i)}
                    disabled={loading || regeneratingIndex !== null}
                  >
                    {regeneratingIndex === i ? '重新生成中...' : '重新生成'}
                  </button>
                  <button
                    className="action-btn ghost"
                    onClick={() => handleCopyResult(i)}
                    disabled={loading}
                  >
                    {copiedIndex === i ? '已复制' : '复制结果'}
                  </button>
                </div>
              </div>
            );
          })}

          {loading && (
            <div className="message system-message">
              <div className="loading-indicator">
                <div className="loading-dots">
                  <span /><span /><span />
                </div>
                <span>
                  {ollamaStatus?.ollama_available
                    ? '正在用 LLM 智能分析你的需求...'
                    : '正在分析你的需求...'}
                </span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </main>
      </div>

      <InputBox onSubmit={handleSubmit} loading={loading} />
    </div>
  );
}
