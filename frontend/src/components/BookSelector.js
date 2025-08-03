import { useEffect, useState } from 'react';
import axios from 'axios';

const BookSelector = ({ setBook }) => {
  const [books, setBooks] = useState([]);

  useEffect(() => {
    axios.get("http://localhost:8007/books/").then(res => setBooks(res.data.books));
  }, []);

  return (
    <select onChange={(e) => setBook(e.target.value)} className="p-2 border rounded">
      <option value="">Select a book</option>
      {books.map(book => (
        <option key={book} value={book}>{book}</option>
      ))}
    </select>
  );
};

export default BookSelector;
