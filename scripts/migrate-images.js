import fs from 'fs';
import path from 'path';
import supabase from '../config/supabase.js';
import pool from '../config/db.js';

async function migrateImages() {
  const imagesDir = './images';
  const metadataFile = './image_data.json';
  const bucketName = 'faces'; // Ensure this exists and is PUBLIC

  try {
    // 1. Load metadata
    if (!fs.existsSync(metadataFile)) {
      console.error("❌ metadata file not found");
      return;
    }
    const metadata = JSON.parse(fs.readFileSync(metadataFile, 'utf8'));

    // 2. Get list of files
    if (!fs.existsSync(imagesDir)) {
      console.error("❌ images directory not found");
      return;
    }
    const files = fs.readdirSync(imagesDir).filter(f => f.match(/\.(jpg|jpeg|png|webp|gif)$/i));

    console.log(`🚀 Starting migration of ${files.length} images...`);

    for (const filename of files) {
      const filePath = path.join(imagesDir, filename);
      const studentInfo = metadata[filename];

      if (!studentInfo) {
        console.warn(`⚠️ No metadata found for ${filename}, skipping or using defaults...`);
      }

      // 3. Upload to Supabase Storage
      const fileBuffer = fs.readFileSync(filePath);
      const { data, error } = await supabase.storage
        .from(bucketName)
        .upload(`students/${filename}`, fileBuffer, {
          upsert: true,
          contentType: 'image/jpeg' // adjust based on ext if needed
        });

      if (error) {
        console.error(`❌ Error uploading ${filename}:`, error.message);
        continue;
      }

      // 4. Get Public URL
      const { data: { publicUrl } } = supabase.storage
        .from(bucketName)
        .getPublicUrl(`students/${filename}`);

      console.log(`✅ Uploaded ${filename} -> ${publicUrl}`);

      // 5. Save to Database
      if (studentInfo) {
        try {
          await pool.query(
            `INSERT INTO students (name, rrn, department, year, section, image_url) 
             VALUES ($1, $2, $3, $4, $5, $6)
             ON CONFLICT (rrn) DO UPDATE SET image_url = EXCLUDED.image_url`,
            [
              studentInfo.Name,
              studentInfo.RRN,
              studentInfo.Department,
              studentInfo.Year,
              studentInfo.Section,
              publicUrl
            ]
          );
          console.log(`💾 Saved metadata for ${studentInfo.Name} to database.`);
        } catch (dbErr) {
          console.error(`❌ DB Error for ${filename}:`, dbErr.message);
        }
      }
    }

    console.log("🎉 Migration complete!");
    process.exit(0);
  } catch (err) {
    console.error("❌ General Error:", err.message);
    process.exit(1);
  }
}

migrateImages();
