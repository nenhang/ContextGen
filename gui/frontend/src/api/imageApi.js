// frontend/src/api/imageApi.js

const BACKEND_BASE_URL = "http://127.0.0.1:5000";

// --- 1. Cutout API Call ---
export async function cutoutImage(files) {
  // 接收文件数组 (files)
  console.log(
    `[API Call] Calling backend cutout API for ${files.length} file(s): ${BACKEND_BASE_URL}/api/cutout`
  );

  const formData = new FormData();

  files.forEach((file) => {
    // 确保这里的键 'files' 与您的后端接收多文件的配置一致
    formData.append("files", file);
  });

  const response = await fetch(`${BACKEND_BASE_URL}/api/cutout`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(`Cutout Failed: ${errorData.error || response.statusText}`);
  }

  const data = await response.json();

  if (!data.cutout_urls || !Array.isArray(data.cutout_urls)) {
    throw new Error("API did not return a list of cutout URLs.");
  }

  // 3. 将相对路径转换为完整的绝对 URL 列表
  const cutoutUrls = data.cutout_urls.map((url) => `${BACKEND_BASE_URL}${url}`);

  console.log(
    `[API Response] Batch Cutout Successful, returned ${cutoutUrls.length} URLs.`
  );

  return cutoutUrls;
}

// --- 2. Model Prediction API Call ---
export async function predictModel(payload) {
  console.log(
    `[API Call] Calling backend model prediction API: ${BACKEND_BASE_URL}/api/predict`
  );

  const response = await fetch(`${BACKEND_BASE_URL}/api/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(
      `Model Prediction Failed: ${errorData.error || response.statusText}`
    );
  }

  const data = await response.json();
  console.log(
    `[API Response] Model Prediction Successful, returned result:`,
    data
  );
  data.result_image_url = `${BACKEND_BASE_URL}${data.result_image_url}`;
  return data;
}

// --- 3. Generate Asset Model API Call ---
export async function generateAssetModel(payload) {
  console.log(
    `[API Call] Calling backend asset generation API: ${BACKEND_BASE_URL}/api/generate_asset`
  );

  const response = await fetch(`${BACKEND_BASE_URL}/api/generate_asset`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(
      `Asset Generation Failed: ${errorData.error || response.statusText}`
    );
  }

  const data = await response.json();
  console.log(
    `[API Response] Asset Generation Successful, returned result:`,
    data
  );
  data.original_url = `${BACKEND_BASE_URL}${data.original_url}`;
  data.cutout_url = `${BACKEND_BASE_URL}${data.cutout_url}`;
  return data;
}
