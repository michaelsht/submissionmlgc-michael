//  Import Packages
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const { Firestore } = require("@google-cloud/firestore");

// Initiation
const app = express();
const port = process.env.PORT || 8080;
const db = new Firestore();
const upload = multer({
  dest: "uploads/",
  limits: { fileSize: 1000000 }, // Limit file size to 1 MB
  //   fileFilter: (req, file, cb) => {
  //     if (file.mimetype !== "image/jpeg") {
  //       cb(new Error("Invalid file type"), false);
  //     } else {
  //       cb(null, true);
  //     }
  //   },
});

// Enable CORS
app.use(cors());

// Routes
app.get("/predict/histories", async (req, res) => {
  // Routes for prediction histories

  const predictCollection = db.collection("predictions");

  const snapshot = await predictCollection.get();

  const data = snapshot.docs.map((doc) => doc.data());

  res.status(200).json({
    status: "success",
    data,
  });
});

app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    // 1. Create Image Uploader
    if (!req.file) {
      return res.status(400).json({
        status: "fail",
        message: "Terjadi kesalahan dalam melakukan prediksi",
      });
    }

    // 2. Success? Convert Image to Tensor
    const buffer = fs.readFileSync(req.file.path);
    const tensor = tf.node
      .decodeJpeg(buffer)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat();

    // 3. Load model
    const model = await tf.loadGraphModel(
      "https://storage.googleapis.com/bucket-cloud-ml-michael/model/model.json"
    );

    // 4. Predict
    const prediction = await model.predict(tensor).data();

    const confidenceScore = Math.max(...prediction) * 100;

    const classes = ["Cancer", "Non-cancer"];
    const classResult = confidenceScore > 50 ? 0 : 1;
    const label = classes[classResult];
    const suggestion =
      classResult == 0
        ? "Cek ke dokter secepatnya!"
        : "Belum ada tanda-tanda tapi tetap jaga kesehatan!";

    // 5. Store Data
    const predictCollection = db.collection("predictions");

    const payload = {
      id: Date.now().toString(),
      result: label,
      suggestion,
      createdAt: new Date(),
    };

    await predictCollection.doc(payload.id).set(payload);

    return res.status(201).json({
      status: "success",
      message: "Model is predicted successfully",
      data: payload,
    });
  } catch (error) {
    return res.status(400).json({
      status: "fail",
      message: "Terjadi kesalahan dalam melakukan prediksi",
    });
  }
});

//Image too large
app.use((err, req, res, next) => {
  if (err.code === "LIMIT_FILE_SIZE") {
    res.status(413).json({
      status: "fail",
      message: "Payload content length greater than maximum allowed: 1000000",
    });
  } else {
    next(err);
  }
});

app.get("/", (req, res) => {
  res.send("Welcome to the submissionmlgc-michaelsihotang Server");
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
