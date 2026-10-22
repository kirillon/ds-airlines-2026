import { useState, type FormEvent } from 'react';
import {
  ArrowRight,
  Eye,
  EyeOff,
  KeyRound,
  LoaderCircle,
  MoveUpRight,
  Route,
  ShieldCheck,
} from 'lucide-react';

type Props = {
  defaultBaseUrl: string;
  onConnect: (baseUrl: string, key: string) => Promise<void>;
};

export default function ConnectionForm({ defaultBaseUrl, onConnect }: Props) {
  const [baseUrl, setBaseUrl] = useState(defaultBaseUrl);
  const [key, setKey] = useState('');
  const [visible, setVisible] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPending(true);
    setError('');
    try {
      await onConnect(baseUrl.trim().replace(/\/+$/, ''), key.trim());
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : 'Не удалось подключиться к API.',
      );
    } finally {
      setPending(false);
    }
  }
  return (
    <main className="connection-page">
      <section className="connection-story" aria-label="О приложении">
        <div className="brand">
          <span className="brand-mark">
            <Route size={22} />
          </span>
          <span>
            Найти шпионов<span className="brand-subtitle">АНАЛИЗ ПЕРЕЛЁТОВ</span>
          </span>
        </div>
        <div className="connection-intro">
          <h1>
            Взгляните
            <br />
            на историю
            <br />
            перелётов <em>шпионов</em>
          </h1>
        </div>
        <div className="connection-art" aria-hidden="true">
          <span className="orbit orbit-one" />
          <span className="orbit orbit-two" />
          <span className="orbit orbit-three" />
          <span className="orbit-point point-one" />
          <span className="orbit-point point-two" />
          <span className="orbit-plane">
            <MoveUpRight size={32} />
          </span>
        </div>
      </section>
      <section className="connection-form-side">
        <form className="connection-form" onSubmit={submit}>
          <span className="form-symbol">
            <KeyRound size={23} />
          </span>
          <span className="eyebrow">РАБОЧЕЕ ПРОСТРАНСТВО</span>
          <h2>Вход по ключу</h2>
          <label htmlFor="api-url">Адрес API</label>
          <input
            id="api-url"
            type="text"
            inputMode="url"
            value={baseUrl}
            onChange={(event) => setBaseUrl(event.target.value)}
            required
            placeholder="http://localhost:8000"
            autoComplete="url"
            disabled={pending}
          />
          <label htmlFor="api-key">API-ключ</label>
          <div className="password-field">
            <input
              id="api-key"
              type={visible ? 'text' : 'password'}
              value={key}
              onChange={(event) => setKey(event.target.value)}
              required
              autoComplete="off"
              spellCheck={false}
              placeholder="Введите ключ доступа"
              disabled={pending}
            />
            <button
              type="button"
              className="icon-button"
              aria-label={visible ? 'Скрыть ключ' : 'Показать ключ'}
              aria-pressed={visible}
              onClick={() => setVisible(!visible)}
            >
              {visible ? <EyeOff size={18} /> : <Eye size={18} />}
            </button>
          </div>
          {error && (
            <div className="error-notice" role="alert">
              {error}
            </div>
          )}
          <button
            type="submit"
            className="button button-primary connect-button"
            disabled={pending || !key.trim()}
          >
            {pending ? (
              <>
                <LoaderCircle className="spinning" size={17} /> Подключаемся…
              </>
            ) : (
              <>
                Войти <ArrowRight size={17} />
              </>
            )}
          </button>
          <p className="privacy-note">
            <ShieldCheck size={17} />
            Ключ хранится только в памяти этой вкладки.
          </p>
        </form>
      </section>
    </main>
  );
}
