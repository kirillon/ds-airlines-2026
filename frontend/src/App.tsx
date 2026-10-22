import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createApi, type Api } from './api';
import ConnectionForm from './components/ConnectionForm';
import Dashboard from './components/Dashboard';

type Session = { api: Api; queryClient: QueryClient };

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  async function connect(baseUrl: string, key: string) {
    const api = createApi(baseUrl, key);
    const tables = await api.tables();
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
          staleTime: 30_000,
          refetchOnWindowFocus: false,
        },
      },
    });
    queryClient.setQueryData(['tables'], tables);
    setSession({ api, queryClient });
  }
  function disconnect() {
    if (session) {
      void session.queryClient.cancelQueries();
      session.queryClient.clear();
    }
    window.history.replaceState(
      null,
      '',
      window.location.pathname + window.location.search,
    );
    setSession(null);
  }
  return session ? (
    <QueryClientProvider client={session.queryClient}>
      <Dashboard api={session.api} onDisconnect={disconnect} />
    </QueryClientProvider>
  ) : (
    <ConnectionForm
      defaultBaseUrl={import.meta.env.VITE_API_URL || '/api'}
      onConnect={connect}
    />
  );
}
