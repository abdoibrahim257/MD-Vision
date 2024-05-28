import React, { useState } from 'react';
import NavBar from './navbar'
import LoadingComponent from './LoadingComponent'
import CaptionGen from './CaptionGen';

import '../styles/ScanPage.css'

import uploadIcon from "../assets/upload.svg"
import bono from "../assets/bono.svg"


const ScanPage = () => {
    const [predictions, setPredictions] = useState([])
    const [predicting, setPredicting] = useState(false)
    const [uploaded, setIsUploaded] = useState(false)

    const handleFileChange = (e) => {
        var f = e.target.files[0]
        // console.log(f.type)
        
        if (!f ){
            setIsUploaded(false)
            return
        }
        // setIsUploaded(true)

        // send the file to back end
        const formData = new FormData()
        formData.append('file', f)
        // console.log(formData)

        const requestOptions = {
            method: 'POST',
            body: formData
        }
        fetch('http://localhost:8000/upload', requestOptions)
            .then(response => {
                if (response.ok) {
                    setIsUploaded(true)
                } else {
                    setIsUploaded(false)
                }
            })
            .catch(error => {
                console.error('There was an error!', error);
            });
    };

    //useeffect for loading screen

    const handlePredict = () => {

        //call the back and get predictions but for now
        setPredicting(true)
        
        //send get request to backend to get predictions
        fetch('http://localhost:8000/upload').then(response => {
            if (response.ok) {
                //convert response to json and set predictions
                response.json().then(data => {
                    console.log(data.message)
                    //add data to list of predictions
                    setPredictions([data.message])
                    // setPredictions(...predictions, data.message)
                    setPredicting(false)
                })
            } else {
                setPredicting(false)
            }}).catch(error => {
                console.error('There was an error!', error);
            });
    }

    return (
        <div>
            { predicting ? <LoadingComponent/> : null}
            <div className='test'>
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
                            <p className='instructions'>1. Upload .png file of your scan</p>
                            <input type="file" id="fileInput" onChange={handleFileChange} accept="image/*" style={{ display: 'none' }} />
                            <button className='upload' onClick={() => document.getElementById('fileInput').click()}>
                                <img className='upIcon' src = {uploadIcon} alt='upload button'/>
                                Click to upload
                            </button>
                            <span className={uploaded ? "success" : "successHidden"}><img src={bono}/>Uploaded Successfully</span>
                        </div>
                        <div className='start-scan'>
                            <p className='instructions'>2. Know your diagnosis</p>
                            <button onClick={handlePredict} className='start'>Start Diagnosing</button>
                        </div>
                        { predictions.length > 0 ? <CaptionGen preds = {predictions}/> : null}
                    </div>
                </div>
            </div>
             
        </div>
  )
}

export default ScanPage