// Import Packages
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const admin = require("firebase-admin");

// Inisialisasi Express App
const app = express();
const port = process.env.PORT || 3000;

// Inisialisasi Firebase Admin SDK dengan Service Account Key
const serviceAccount = require("./submissionmlgc-michaelsi-1b3b4-firebase-adminsdk-mrokh-799b96c52a.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: "https://submissionmlgc-michaelsi-1b3b4.firebaseio.com",
});

// Mendapatkan instance Firestore
const db = admin.firestore();

// Setup Multer untuk Upload File
const upload = multer({
  dest: "uploads/",
  limits: { fileSize: 1000000 }, // Limit file size to 1 MB
});

// Enable CORS
app.use(cors());

// Route untuk Melihat Riwayat Prediksi
app.get("/predict/histories", async (req, res) => {
  try {
    const snapshot = await db.collection("predictions").get();
    const data = snapshot.docs.map((doc) => doc.data());

    res.status(200).json({
      status: "success",
      data,
    });
  } catch (error) {
    console.error("Error fetching prediction history:", error);
    res.status(500).json({
      status: "fail",
      message: "Gagal mengambil riwayat prediksi",
    });
  }
});

// Route untuk Melakukan Prediksi
app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    // 1. Cek jika file di-upload
    if (!req.file) {
      return res.status(400).json({
        status: "fail",
        message: "Terjadi kesalahan dalam melakukan prediksi, file tidak ditemukan",
      });
    }

    // 2. Convert Image to Tensor
    const buffer = fs.readFileSync(req.file.path);
    const tensor = tf.node
      .decodeJpeg(buffer)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat();

    // 3. Load Model untuk Prediksi
    const model = await tf.loadGraphModel(
      "https://storage.googleapis.com/bucket-cloud-ml-michael/model/model.json"
    );

    // 4. Lakukan Prediksi
    const prediction = await model.predict(tensor).data();
    const confidenceScore = Math.max(...prediction) * 100;

    const classes = ["Cancer", "Non-cancer"];
    const classResult = confidenceScore > 50 ? 0 : 1;
    const label = classes[classResult];
    const suggestion =
      classResult === 0
        ? "Cek ke dokter secepatnya!"
        : "Belum ada tanda-tanda tapi tetap jaga kesehatan!";

    // 5. Simpan Data Prediksi ke Firestore
    const payload = {
      id: Date.now().toString(),
      result: label,
      suggestion,
      createdAt: new Date(),
    };

    await db.collection("predictions").doc(payload.id).set(payload);

    // Kirimkan Response ke Pengguna
    return res.status(201).json({
      status: "success",
      message: "Prediksi berhasil dilakukan",
      data: payload,
    });
  } catch (error) {
    console.error("Error during prediction:", error);
    return res.status(500).json({
      status: "fail",
      message: "Terjadi kesalahan dalam melakukan prediksi",
    });
  }
});

// Middleware untuk Menangani Error Ukuran File Terlalu Besar
app.use((err, req, res, next) => {
  if (err.code === "LIMIT_FILE_SIZE") {
    res.status(413).json({
      status: "fail",
      message: "Ukuran file melebihi batas yang diperbolehkan (1MB)",
    });
  } else {
    next(err);
  }
});

// Route Default
app.get("/", (req, res) => {
  res.send("Selamat datang di server prediksi kanker!");
});

// Menjalankan Server
app.listen(port, () => {
  console.log(`Server berjalan di port ${port}`);
});
