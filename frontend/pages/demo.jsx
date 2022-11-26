import Image from 'next/image'
import { useState } from "react";

export default function Demo() {
    const demo_list = [
        {
            src: '/adriaen-brouwer_feeling.jpg', 
            attention_list: ['attention1-1', 'attention1-2'],
            attention_impression: [
                {
                    attention: 'attention1-1',
                    impression: 'impression1-1'
                },
                {
                    attention: 'attention1-2',
                    impression: 'impression1-2'
                },
            ]
        },
        {
            src: '/adriaen-van-ostade_smoker.jpg', 
            attention_list: ['attention2-1', 'attention2-2'],
            attention_impression: [
                {
                    attention: 'attention2-1',
                    impression: 'impression2-1'
                },
                {
                    attention: 'attention2-2',
                    impression: 'impression2-2'
                },
            ]
        },
        {
            src: '/albert-bloch_piping-pierrot.jpg', 
            attention_list: ['attention3-1', 'attention3-2'],
            attention_impression: [
                {
                    attention: 'attention3-1',
                    impression: 'impression3-1'
                },
                {
                    attention: 'attention3-2',
                    impression: 'impression3-2'
                },
            ]
        },
        {
            src: '/chuck-close_self-portrait-2000.jpg', 
            attention_list: ['attention4-1', 'attention4-2'],
            attention_impression: [
                {
                    attention: 'attention4-1',
                    impression: 'impressio4-1'
                },
                {
                    attention: 'attention4-2',
                    impression: 'impression4-2'
                },
            ]
        },
        {
            src: '/martiros-saryan_still-life-1913.jpg', 
            attention_list: ['attention5-1', 'attention5-2'],
            attention_impression: [
                {
                    attention: 'attention5-1',
                    impression: 'impression5-1'
                },
                {
                    attention: 'attention5-2',
                    impression: 'impression5-2'
                },
            ]
        },
    ]

    const [src, setSrc] = useState()
    const [attentionList, setAttentionList] = useState([])
    const [impression, setImpression] = useState('')
    const [checked, setChecked] = useState('')

    const changeSrc = e => {
        setSrc(e.target.value);
        setAttentionList(demo_list.find(value => value.src == e.target.value).attention_list)
        setChecked('')
        setImpression('')
    }

    const changeAttention = e => {
        setChecked(e.target.value)
        setImpression(demo_list.find(value => value.src == src).attention_impression.map((element) => {
            if (element.attention == e.target.value) {
                return element.impression
            }
        }))
    }

    return (
        <div>
            <h1 className='font-bold text-4xl text-center pb-10'>Demo</h1>

            <div>
                <ul className='grid grid-cols-5 place-items-center pb-10'>
                    {demo_list.map((demo, idx) => {
                        return (
                            <li key={idx}>
                                <Image src={demo.src} objectFit='contain' width={180} height={180} alt='test' />
                                <div className='text-center'>
                                    <input type='radio' name='radio-img' value={demo.src} className='mt-2' onChange={changeSrc} />
                                </div>
                            </li>
                        )
                    })}
                </ul>
            </div>
            
            <div className='grid grid-cols-2 place-items-center pb-10 w-1/3 mx-auto'>
                <div>
                    <p className='font-bold text-2xl'>着目点を選択→</p>
                </div>

                <div>
                    <ul>
                        {attentionList.map((attention, idx) => {
                            return (
                                <li key={idx}>
                                    <input type='radio' name='radio-attention' value={attention} onChange={changeAttention} checked={checked === attention} />
                                    <label>{attention}</label>
                                </li>
                            )
                        })}
                    </ul>
                </div>
            </div>
            
            <div>
                <p className='text-center text-2xl'>感想:{impression}</p>
            </div>

        </div>
    )
}