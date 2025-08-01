import axios from 'axios';

const API_BASE = 'http://localhost:8006';

export const fetchBooks = async () => {
  const response = await axios.get(`${API_BASE}/books/`);
  return response.data.books;
};

export const sendQuery = async (query, sessionId, bookName) => {
  const formData = new FormData();
  formData.append('query', query);
  formData.append('session_id', sessionId);
  formData.append('book_name', bookName);

  const response = await axios.post(`${API_BASE}/query/`, formData);
  return response.data;
};
