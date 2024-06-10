import React, { useEffect, useReducer , useRef} from 'react'
import '../styles/Chat.css'
import '../styles/heroSection.css'
import NavBar from './navbar.js'
import robot from '../assets/loveBot.svg'
import warning from '../assets/WARNING.png'
// eslint-disable-next-line
import * as LottiePlayer from "@lottiefiles/lottie-player";

import {
  MainContainer,
  ChatContainer,
  MessageList,
  Message
} from "@chatscope/chat-ui-kit-react";

const initialState = { index: 0 };

function reducer(state, action) {
  switch (action.type) {
    case 'increment':
      return { index: state.index + 1 };
    // Add more cases for other actions
    default:
      throw new Error();
  }
}

const ChatPage = () => {
  // const [padded, setPadded] = React.useState(false)
  const [state, dispatch] = useReducer(reducer, initialState);
  const incrementIndex = () => {
    dispatch({ type: 'increment' });
  };

  
  // const [messages, setMessages] = React.useState([])
  const [messages, setMessages] = React.useState({})

  //   message: "Hello, I'm Maven, your personal health assistant. How can I help you today?",
  //   sender: 'Maven'
  
  const [diagnosed, setDiagnosed] = React.useState(false)

  const currentLink = window.location.href

  const bottomRef = useRef();



  const typeWriter = (key, msg, msgIndex) => {
    setTimeout(() => {
      // Update the state
      setMessages((oldMessages) => {
        // Create a copy of the messages object
        var newMessages = {...oldMessages};
        // Update the message at the given key
        newMessages[key] = {...newMessages[key], message: msg.substr(0, msgIndex)};
        return newMessages;
      });
      // If there are more characters in the message, call typeWritter again
      if (msgIndex < msg.length) {
        typeWriter(key, msg, msgIndex + 1);
      }
    }, 23);
  };

const fetchFirstMessage = async () => {
    let symptom = currentLink.split('/').pop()

    const response = await fetch('https://shad-honest-anchovy.ngrok-free.app/maven/'+symptom , {
        headers: new Headers({
            "ngrok-skip-browser-warning": "69420",
        }),
    })

    const data = await response.json()

    const message = {
        // message: data.Question,
        message: "",
        sender: 'Maven'
    }

    setMessages((oldMessages = {}) => {
        var newMessages = {...oldMessages};
        let newKey = state.index; // get the next key
        // console.log("FIRST: ",newKey)
        newMessages[newKey] = message;
        return newMessages;
    });

    typeWriter(state.index, data.Question, 0);
    incrementIndex();
}

  useEffect(() => {
    fetchFirstMessage()
    // eslint-disable-next-line
  }, [])
  

  async function fetchNextQuestion (answer) {
    let symptom = currentLink.split('/').pop();
    let url = 'https://shad-honest-anchovy.ngrok-free.app/maven/'+symptom;

    fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json' //specify the content type
        // 'ngrok-skip-browser-warning': '69420',
      },
      body: JSON.stringify({ans: answer}),
    }).then(response => response.json())
    .then(data => {
      const Q = data.Question
      // check if the word Diagnosis: is in the question
      if (Q.includes('Diagnosis:')) {
        setDiagnosed(true)
      }

      const m = {
        // message: Q,
        message: "",
        sender: 'Maven'
      }

      setMessages((oldMessages = {}) => {
        var newMessages = {...oldMessages};
        let newKey = state.index 
        console.log("HANDLE NEXT Q: ",newKey)
        newMessages[newKey+1] = m;
        return newMessages;
      });

      // console.log(messages)
      typeWriter(state.index+1, Q,0)
      incrementIndex();
    })
}

  const handleYes = async () => {
    const newMessage = {
      message: "Yes",
      sender: 'user',
      direction: "outgoing"
    }
    
    setMessages((oldMessages = {}) => {
      var newMessages = {...oldMessages};
      let newKey = state.index;
      console.log(newKey)
      newMessages[newKey-1] = {...newMessages[newKey-1], status: "answered"};
      newMessages[newKey] = newMessage;
      console.log(newMessages)

      return newMessages;
    })

    incrementIndex();
    await fetchNextQuestion('yes')

    bottomRef.current.scrollIntoView({ behavior: "smooth" });
    
  }

  const handleNo = async () => {

    const newMessage = {
      message: "No",
      sender: 'user',
      direction: "outgoing"
    }
  
    setMessages((oldMessages = {}) => {
      var newMessages = {...oldMessages};
      let newKey = state.index; 
      console.log(newKey)
      newMessages[newKey-1] = {...newMessages[newKey-1], status: "answered"};
      newMessages[newKey] = newMessage;
      console.log(newMessages)

      return newMessages;
    })

    incrementIndex();
    await fetchNextQuestion('no')

    bottomRef.current.scrollIntoView({ behavior: "smooth" });
  }

  return (
    <div>
        <NavBar sticky={1} />
        <div className='chat-section content'>
          <div  className='chat-content'>
            <lottie-player
              autoplay
              loop
              mode="normal"
              // src="https://lottie.host/d73b52cc-4991-44cf-b125-45e96577b4bc/or62hvWNOo.json"
              src="https://lottie.host/3ad60a0d-fcbc-444f-9431-e861fe6a2368/zYg28wXoYO.json"
            ></lottie-player>

            <div className='warning-wrapper'>
              <img className='warning-img' width="40px" height="40px" src={warning} alt='Warning logo' />
              <div className='warning-message'>
                <p> This tool is not a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing a life-threatening emergency that requires immediate attention please call 123 or the number for your local emergency service.</p>
              </div>
            </div>
          </div>

          <div className='interactive-section'>
            <MainContainer>
              <ChatContainer>
                <MessageList>
                    {
                      Object.keys(messages).map((key) => {
                      const message = messages[key];
      
                      if (message.sender === 'Maven' && !message.status) {
                        return (
                          <div key={key}>
                            <div className='message-wrapper'>
                              <img src={robot} alt='robot' className='robot'/>
                              <Message className = "message" model={message}/>
                            </div>
                            <div className={diagnosed? 'Hidden' : 'answer-wrapper'}>
                              <button className='btn yes' onClick={handleYes}>Yes</button>
                              <button className='btn no' onClick={handleNo}>No</button>
                            </div>
                            
                          </div>
                        );
                      }
                      else if (message.status && message.status === "answered" && message.sender === 'Maven') {
                        return (
                            <div key={key} className='message-wrapper'>
                              <img src={robot} alt='robot' className='robot'/>
                              <Message className = "message" model={message}/>
                            </div>
                        );
                      }
                      else {
                        return (
                          <div>
                            <Message className = "answer-message" key={key} model={message}/>
                          </div>
                        );
                      }
                    })
                    }
                </MessageList>
              </ChatContainer>
            </MainContainer>
          </div>
          <div ref={bottomRef}></div>
        </div>
  
    </div>
  )
}

export default ChatPage