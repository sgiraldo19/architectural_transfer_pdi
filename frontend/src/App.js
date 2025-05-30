import React, { useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [style, setStyle] = useState("barroco");
  const [generatedImage, setGeneratedImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setGeneratedImage(null);
    if (file) {
      setPreviewURL(URL.createObjectURL(file));
    }
  };

  const handleStyleChange = (e) => {
    setStyle(e.target.value);
  };

  const handleSubmit = async () => {
    if (!image) {
      alert("Por favor, selecciona una imagen.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("image", image);
    formData.append("style", style);

    try {
      const response = await fetch("http://localhost:5000/generate", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Error al generar imagen.");

      const blob = await response.blob();
      setGeneratedImage(URL.createObjectURL(blob));
    } catch (error) {
      console.error(error);
      alert("Ocurrió un error al generar la imagen.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1 className="app-title">Transformador de Estilo Arquitectónico</h1>

      <div className="upload-panel">
        <input type="file" onChange={handleImageChange} />
        <select className="select-style" value={style} onChange={handleStyleChange}>
          <option value="barroco">Convertir al estilo Barroco</option>
          <option value="gotico">Convertir al estilo Gótico</option>
        </select>
        <button className="upload-button" onClick={handleSubmit} disabled={loading}>
          {loading ? "Generando..." : "Transformar"}
        </button>
      </div>

      {(previewURL || generatedImage) && (
        <div className="image-preview">
          {previewURL && (
            <div className="image-column">
              <p className="caption">Original</p>
              <img src={previewURL} alt="Imagen original" />
            </div>
          )}
          {generatedImage && (
            <div className="image-column">
              <p className="caption">Transformada</p>
              <img src={generatedImage} alt="Imagen generada" />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
