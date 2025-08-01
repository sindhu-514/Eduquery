import { Document, Page } from 'react-pdf';
import { useState } from 'react';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';

const BookViewer = ({ book }) => {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);

  const onDocumentLoadSuccess = ({ numPages }) => setNumPages(numPages);

  if (!book) return <div className="text-center mt-8">Select a book to start reading</div>;

  return (
    <div className="w-full h-full flex flex-col items-center justify-center">
      <Document
        file={`/books/${book}.pdf`}
        onLoadSuccess={onDocumentLoadSuccess}
        className="shadow-md"
      >
        <Page pageNumber={pageNumber} width={600} />
      </Document>
      <div className="mt-4 flex gap-2">
        <button onClick={() => setPageNumber(p => Math.max(p - 1, 1))}>◀</button>
        <span>{pageNumber} / {numPages}</span>
        <button onClick={() => setPageNumber(p => Math.min(p + 1, numPages))}>▶</button>
      </div>
    </div>
  );
};

export default BookViewer;
