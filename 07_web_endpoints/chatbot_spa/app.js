// Source code for a single-page app using the Solid.js framework.

import {
  children,
  createEffect,
  createSignal,
  For,
  Show,
} from "https://cdn.skypack.dev/solid-js";
import { render } from "https://cdn.skypack.dev/solid-js/web";
import html from "https://cdn.skypack.dev/solid-js/html";

async function sendMessage(message, id) {
  const resp = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, id }),
  });
  if (resp.status !== 200) {
    throw new Error("An error occurred: " + resp.status);
  }
  return await resp.json();
}

function Layout(props) {
  const c = children(() => props.children);
  return html`
    <div class="absolute inset-0 bg-gray-50 px-2">
      <div class="mx-auto max-w-md py-8 sm:py-16">
        <main class="rounded-xl bg-white p-4 shadow-lg">
          <h1 class="text-center text-2xl font-semibold">
            Talk to${" "}
            <a href="https://modal.com" class="text-lime-700">Modal</a>${" "}
            Transformer
          </h1>
          ${c}
        </main>
      </div>
    </div>
  `;
}

function App() {
  const [id, setId] = createSignal(null);
  const [message, setMessage] = createSignal("");
  const [sending, setSending] = createSignal(false);
  const [chat, setChat] = createSignal(["Hello! I'm a bot running on Modal."]);

  let chatEl;

  // Scroll to bottom when new messages appear.
  createEffect(() => {
    chat();
    chatEl.scrollTop = chatEl.scrollHeight;
  });

  async function handleSubmit(event) {
    event.preventDefault();
    setSending(true);
    try {
      setChat([...chat(), message()]);
      const resp = await sendMessage(message(), id());
      setChat([...chat(), resp.response]);
      setId(resp.id);
      setMessage("");
    } catch (error) {
      alert("An error occurred: " + error.message);
      setChat(chat().slice(0, -1));
    } finally {
      setSending(false);
    }
  }

  return html`
    <${Layout}>
      <div
        ref=${(el) => (chatEl = el)}
        class="h-[320px] overflow-x-hidden overflow-y-scroll space-y-2 my-4"
      >
        <${For} each=${chat}>
          ${(msg, i) => html`
            <div class=${"flex " + (i() % 2 ? "justify-end" : "justify-start")}>
              <div
                class=${"rounded-[16px] px-3 py-1.5 " +
                (i() % 2
                  ? "bg-indigo-500 text-white ml-8"
                  : "bg-gray-100 mr-8")}
              >
                ${msg}
              </div>
            </div>
          `}
        <//>

        <${Show} when=${sending}>
          <div class="flex justify-start animate-pulse">
            <div
              class="bg-gray-100 text-gray-500 select-none rounded-[16px] px-3 py-1.5"
            >
              •••
            </div>
          </div>
        <//>
      </div>

      <form onSubmit=${handleSubmit}>
        <input
          value=${message}
          onInput=${(e) => setMessage(e.target.value)}
          class="w-full rounded-[16px] border px-3 py-1.5 disabled:opacity-50"
          placeholder="Aa"
          autofocus
          required
          disabled=${sending}
        />
      </form>
    <//>
  `;
}

render(App, document.body);
