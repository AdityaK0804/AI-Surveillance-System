import express from "express";

const router=express.Router();


router.get("/",(req,res)=>{
    res.render("contact")
})

router.post("/", (req, res) => {
    const { name, email, message } = req.body;
  
    // Process the form data (for example, save to a database or send an email)
    // For now, we'll just log the data to the console
    console.log(`Name: ${name}, Email: ${email}, Message: ${message}`);
  
    // Respond with a success message or redirect to another page
     res.render('contact',{
        loginsucess: '✅Thank you for contacting us.'
     })
  });

export default router;
