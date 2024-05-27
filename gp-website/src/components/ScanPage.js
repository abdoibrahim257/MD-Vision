import React, { useState, useEffect } from 'react';
import NavBar from './navbar'
import '../styles/ScanPage.css'
import uploadIcon from "../assets/upload.svg"
import bono from "../assets/bono.svg"


const ScanPage = () => {

    const [file, setFile] = useState('')
    const [uploaded, setIsUploaded] = useState(false)

    const handleFileChange = (e) => {
        var file = e.target.files[0]
        if (file && file.type.startsWith('image/')){
            setIsUploaded(true)
            setFile(e.target.files[0])
        }else{
            setIsUploaded(false)
        }
    
        console.log(e.target.files[0])
    };

    return (
        <div>
            <div class="scan-background">
                <NavBar/>
                <div class="content">
                    <div className='hero-wrapper'>
                        <div className='page-quote'>
                            <p><span id='upload'>Upload</span> a scan</p>
                            <p><span id='ease'>Ease</span> your mind</p>
                            <p>Let's keep your health journey <span id='free'>worry-free!</span></p>
                        </div>
                        <div className='lottie'>
                            <lottie-player
                                autoplay
                                mode="normal"
                                speed = "1.8"
                                src="https://lottie.host/b911b2fd-06fd-4d06-86f7-beb19feb7a6b/oLUWaaFmk1.json"
                            ></lottie-player>
                        </div>
                    </div>
                </div>
            </div>
            <div className='content'>
                <div class="scan-wrapper">
                    <p className='objective'>Use AI to Diagnose for your scan.</p>
                    <div class="scan">
                        <input type="file" id="fileInput" onChange={handleFileChange} accept="image/*" style={{ display: 'none' }} />
                        <button className='upload' onClick={() => document.getElementById('fileInput').click()}>
                            <img className='upIcon' src = {uploadIcon} alt='upload button'/>
                            Upload the scan
                        </button>
                        <span className={uploaded ? "success" : "successHidden"}><img src={bono}/>Uploaded Successfully</span>
                    </div>
                    <button></button>
                </div>
            </div>                
        </div>
  )
}

export default ScanPage