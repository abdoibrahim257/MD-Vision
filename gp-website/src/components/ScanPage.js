import React, { useState } from 'react';
import NavBar from './navbar'
import LoadingComponent from './LoadingComponent'
import CaptionGen from './CaptionGen';
import Swal from 'sweetalert2';

import '../styles/ScanPage.css'

import uploadIcon from "../assets/upload.svg"
import bono from "../assets/bono.svg"
import reload from "../assets/reload.svg"


const ScanPage = () => {
    const [model, setModel] = useState('')
    const [initType, setInitialisedType] = useState('')
    const [padded, setPadded] = useState(false)
    const [predictions, setPredictions] = useState([])
    const [predicting, setPredicting] = useState(false)
    const [uploaded, setIsUploaded] = useState(false)
    const [uploadedImage, setUploadedImage] = useState('')


    const handleInitializeModel = (e) => {
        setInitialisedType(e.target.value)
        let initType = e.target.value
        if (initType !== ""){
            fetch(`http://127.0.0.1:8000/upload/?init=0&model=${initType}`)
            .then(response => {
                if (response.ok) {
                    console.log("model initialised")
                } else {
                    console.log("model not initialised")
                }
            }).catch(error => {
                console.error('There was an error!', error);
            });
        }
    }


    const handleModelChange = (e) => {
        setModel(e.target.value)
    };

    const handleFileChange = (e) => {
        var f = e.target.files[0]

        if (!f || f.type !== 'image/png') {
            setIsUploaded(false)
            return
        }

        // send the file to back end
        const formData = new FormData()
        formData.append('file', f)

        const requestOptions = {
            method: 'POST',
            body: formData
        }
        // fetch('https://shad-honest-anchovy.ngrok-free.app/upload', requestOptions)
        fetch('http://127.0.0.1:8000/upload', requestOptions)
        .then(response => {
            if (response.ok) {
                setIsUploaded(true)
                // show the image we uploaded here
                setUploadedImage(URL.createObjectURL(formData.get('file')))
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
        if (initType !== ""){
            if (model === "coAtt" || model === "kengic") {
    
                setPredicting(true)
        
                // fetch('https://shad-honest-anchovy.ngrok-free.app/upload', {
                //     headers: new Headers({
                //         "ngrok-skip-browser-warning": "69420",
                //     }),
                // }).then(response => {
                fetch(`http://127.0.0.1:8000/upload/?init=1&model=${model}`).then(response => {
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
                    }
                }).catch(error => {
                    console.error('There was an error!', error);
                    setPredicting(false)
                    Swal.fire({
                        icon: 'error',
                        title: 'Oops...',
                        text: 'An error occured while diagnosing your scan! Please try again later.',
                    })
                });
            }
            else{
                Swal.fire({
                    icon: 'error',
                    title: 'Oops...',
                    text: 'Please select a model to start diagnosing!',
                })
            }
        }
        else{
            Swal.fire({
                icon: 'error',
                title: 'Oops...',
                text: 'Please select which Visual Extractor to use First!',
            })
        }
    };

    return (
        <div>
            {predicting ? <LoadingComponent /> : null}
            <div className='test'>
                <div className="scan-background">
                    <NavBar setPadding={setPadded}/>
                    <div className={padded ? "content maintain-content" : "content"}>
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
                                    speed="1.8"
                                    src="https://lottie.host/b911b2fd-06fd-4d06-86f7-beb19feb7a6b/oLUWaaFmk1.json"
                                ></lottie-player>
                            </div>
                        </div>
                    </div>
                </div>
                <div className='content'>
                    <div class="scan-wrapper">
                        <p className='objective'>Use AI to Diagnose your scan.</p>
                        <div class="scan">
                            <div className='nashat'>
                                <p className='instructions'>1. Upload .png file of your scan</p>
                                {uploaded && 
                                    <img className='reupload' onClick={() => document.getElementById('fileInput').click()} src={reload} alt='reupload' />
                                }
                            </div>
                            <input type="file" id="fileInput" onChange={handleFileChange} accept="image/png" style={{ display: 'none' }} />
                            {
                                uploaded ?
                                    <div className='divUploaded'>
                                        <img className='uploaded-image' src={uploadedImage} alt='uploaded' />
                                    </div>
                                    :
                                    <button className='upload' onClick={() => document.getElementById('fileInput').click()} disabled={uploaded}>
                                        <div>
                                            <img className='upIcon' src={uploadIcon} alt='upload button' />
                                            Click to upload
                                        </div>
                                    </button>
                            }
                            <span className={uploaded ? "success" : "successHidden"}><img src={bono} alt='successfull upload'/>Uploaded Successfully</span>
                        </div>
                        <div className="model-type">
                            <p className='instructions'>2. Choose the Visual Extractor to use</p>
                            <select className='DropDown' onChange={handleInitializeModel}>
                                <option value = "" disabled selected> Select a model</option>
                                <option value = "densenet121">DenseNet121</option>
                                <option value = "hog_pca">Hog&PCA</option>
                                <option value = "resnet50">ResNet50</option>
                                <option value = "resnet152">ResNet152</option>
                            </select>
                        </div>
                        <div className="model-type">
                            <p className='instructions'>3. Choose the caption Generation model to use</p>
                            <select className='DropDown' onChange={handleModelChange}>
                                <option value = "" disabled selected> Select a model</option>
                                <option value = "kengic">Kengic</option>
                                <option value = "coAtt">LSTM</option>
                            </select>
                        </div>
                        <div className={uploaded? 'start-uploaded' : 'start-scan'}>
                            <p className='instructions'>4. Know your diagnosis</p>
                            <button onClick={handlePredict} className='start' disabled={!uploaded}>Start Diagnosing</button>
                        </div>
                        {predictions.length > 0 ? <CaptionGen preds={predictions} /> : null}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default ScanPage