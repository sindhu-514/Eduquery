import { useState } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

const ChatBot = ({ book }) => {
  const [input, setInput] = useState('');
  const [chat, setChat] = useState([]);
  const sessionId = uuidv4();
6
  const sendMessage = async () => {
    if (!input) return;

    const form = new FormData();
    form.append("query", input);
    form.append("session_id", sessionId);
    form.append("book_name", book);

    const res = await axios.post("http://localhost:8007/query/", form);
    const aiReply = res.data.answer;

    setChat(prev => [...prev, { user: input, ai: aiReply }]);
    setInput('');
  };

  return (
    <div className="h-full p-4 flex flex-col justify-between border-l">
      <div className="overflow-y-auto mb-4 space-y-2">
        {chat.map((entry, i) => (
          <div key={i}>
            <p><strong>User:</strong> {entry.user}</p>
            <p><strong>AI:</strong> {entry.ai}</p>
          </div>
        ))}
      </div>
      <div className="flex gap-2">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          className="flex-1 p-2 border rounded"
          placeholder="Ask something..."
        />
        <button onClick={sendMessage} className="p-2 bg-blue-500 text-white rounded">Send</button>
      </div>
    </div>
  );
};

export default ChatBot;
